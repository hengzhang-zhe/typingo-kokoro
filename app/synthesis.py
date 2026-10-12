"""Validated task controls shared by import and generation."""
import hashlib
import math
import re
from decimal import Decimal
from app.pronunciation import ipa_to_misaki


def speed_string(value):
    try:
        if isinstance(value, bool):
            raise ValueError()
        numeric = float(value)
        if not math.isfinite(numeric) or not 0.5 <= numeric <= 2:
            raise ValueError()
        result = format(Decimal(str(value)).normalize(), 'f')
        return result if '.' in result else result + '.0'
    except (ValueError, TypeError, ArithmeticError):
        raise ValueError('Speed must be between 0.5 and 2.0') from None


def validate_options(item, required=False):
    options = item.get('synthesis')
    if options is None:
        if required:
            raise ValueError('Explicit synthesis options are required')
        options = dict(mode='ipa' if item.get('pronunciationId') else 'text', speed='1.0', overrides=[])
    if not isinstance(options, dict) or set(options) != {'mode', 'speed', 'overrides'}:
        raise ValueError('Invalid synthesis options')
    speed = speed_string(options['speed'])
    overrides = options['overrides']
    if not isinstance(overrides, list):
        raise ValueError('Pronunciation overrides must be an array')
    expected_mode = 'ipa' if item.get('pronunciationId') else 'mixed' if overrides else 'text'
    if options['mode'] != expected_mode or (expected_mode == 'ipa' and overrides):
        raise ValueError('Synthesis mode differs from pronunciation target')
    ends = {}
    for o in overrides:
        if not isinstance(o, dict) or set(o) != {'start', 'end', 'text', 'ipa', 'locale'}:
            raise ValueError('Invalid pronunciation override')
        start, end, locale = o['start'], o['end'], o['locale']
        if type(start) is not int or type(end) is not int or locale not in {'en-US', 'en-GB'} or not 0 <= start < end <= len(item['text'].encode('utf-16-le')) // 2 or start < ends.get(locale, 0):
            raise ValueError('Invalid or overlapping pronunciation override offsets')
        try:
            actual = item['text'].encode('utf-16-le')[start * 2:end * 2].decode('utf-16-le')
        except UnicodeDecodeError:
            raise ValueError('Pronunciation override splits a UTF-16 surrogate pair') from None
        if actual != o['text']:
            raise ValueError('Pronunciation override text differs from source span')
        ipa_to_misaki(o['ipa'], locale)
        ends[locale] = end
    return dict(mode=expected_mode, speed=speed, overrides=overrides)



def override_tokens(text, tokens, overrides, locale):
    """Preserve contextual G2P; replace only an exact whole-token span."""
    wanted = {(o['start'], o['end']): o for o in overrides if o['locale'] == locale}
    cursor = 0
    for token in tokens:
        start = text.find(token.text, cursor)
        if start < 0 or text[cursor:start].strip():
            raise ValueError('Cannot align contextual tokens with source text')
        end = start + len(token.text)
        span = (len(text[:start].encode('utf-16-le')) // 2, len(text[:end].encode('utf-16-le')) // 2)
        override = wanted.pop(span, None)
        if override:
            token.phonemes = ipa_to_misaki(override['ipa'], locale)
        cursor = end
    if wanted:
        raise ValueError('Pronunciation override must match one complete contextual token')
    return tokens


def for_voice(options, locale):
    overrides = [o for o in options['overrides'] if o['locale'] == locale]
    return dict(mode='ipa' if options['mode'] == 'ipa' else 'mixed' if overrides else 'text',
                speed=options['speed'], overrides=overrides)


def generate_contextual(pipeline, text, overrides, locale, voice, speed, model):
    """Keep the text pipeline's newline boundaries while retaining sentence context."""
    for match in re.finditer(r'[^\n]+', text):
        segment = match.group()
        if not segment.strip():
            continue
        base = len(text[:match.start()].encode('utf-16-le')) // 2
        limit = base + len(segment.encode('utf-16-le')) // 2
        selected = []
        for o in overrides:
            if o['locale'] != locale:
                continue
            if o['start'] < limit and o['end'] > base:
                if o['start'] < base or o['end'] > limit:
                    raise ValueError('Pronunciation override crosses a newline boundary')
                selected.append(dict(o, start=o['start'] - base, end=o['end'] - base))
        _, tokens = pipeline.g2p(segment)
        tokens = override_tokens(segment, tokens, selected, locale)
        yield from pipeline.generate_from_tokens(tokens, voice=voice, speed=speed, model=model)

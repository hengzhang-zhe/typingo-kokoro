"""Strict English dictionary IPA adapter; never guess unknown symbols."""
import re

def ipa_to_misaki(ipa: str, locale: str) -> str:
    if locale not in {'en-US', 'en-GB'}:
        raise ValueError('Unsupported pronunciation locale')
    if not isinstance(ipa, str) or not ipa.strip() or ipa != ipa.strip() or any(c in ipa for c in '/[]()') or any(c.isspace() for c in ipa):
        raise ValueError('Expected definite IPA without delimiters')
    value = ipa.replace('.', '').replace('g', 'ɡ')
    for old, new in [('tʃ', 'ʧ'), ('dʒ', 'ʤ'), ('eɪ', 'A'), ('aɪ', 'I'), ('aʊ', 'W'), ('ɔɪ', 'Y'),
                     ('oʊ', 'O'), ('əʊ', 'Q'), ('ɚ', 'əɹ'), ('ɝ', 'ɜɹ'), ('r', 'ɹ'), ('e', 'ɛ')]:
        value = value.replace(old, new)
    if locale == 'en-US':
        value = value.replace('ː', '')
    else:
        value = value.replace('æ', 'a')
    vocab = set('AIWYbdfhijklmnpstuvwzðŋɑɔəɛɜɡɪɹʃʊʌʒʤʧˈˌθᵊ ' + ('Qaɒː' if locale == 'en-GB' else 'Oæɾᵻ'))
    if not value or any(c not in vocab for c in value):
        raise ValueError('Unsupported or accent-incompatible IPA; pronunciation needs review')
    value = re.sub(r'([ˈˌ])([bdfhjklmnpstvwzðŋɡɹʃʒʤʧθ]+)', r'\2\1', value)
    if len(value) > 510:
        raise ValueError('Pronunciation exceeds model limit')
    return value

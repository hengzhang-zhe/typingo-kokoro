"""Voice-independent, versioned Misaki G2P. Never converts dictionary IPA."""
import hashlib
import json
import re
import threading
from functools import lru_cache
from importlib import metadata
from pathlib import Path

_lock = threading.RLock()

@lru_cache(maxsize=1)
def g2p_version():
    import misaki
    import kokoro
    versions = {name: metadata.version(name) for name in ('misaki', 'kokoro', 'spacy', 'en-core-web-sm')}
    digest = hashlib.sha256(Path(__file__).read_bytes())
    for package in (misaki, kokoro):
        for path in sorted(Path(package.__file__).parent.rglob('*')):
            if path.is_file() and path.suffix in {'.py', '.json', '.txt'}:
                digest.update(path.relative_to(Path(package.__file__).parent).as_posix().encode())
                digest.update(path.read_bytes())
    # Fallback availability is part of the actual G2P configuration.
    from misaki import espeak
    try:
        fallback = espeak.EspeakFallback(british=False)
        versions['fallback'] = 'espeak'
        import espeakng_loader
        library = Path(espeakng_loader.get_library_path())
        digest.update(library.read_bytes())
    except Exception:
        versions['fallback'] = 'disabled'
    return 'misaki:' + json.dumps(versions, sort_keys=True, separators=(',', ':')) + ':' + digest.hexdigest()

@lru_cache(maxsize=2)
def pipeline(locale):
    if locale not in {'en-US', 'en-GB'}:
        raise ValueError('Only en-US and en-GB are supported')
    from kokoro import KPipeline
    return KPipeline(lang_code='a' if locale == 'en-US' else 'b', repo_id='hexgrad/Kokoro-82M', model=False)

def phonemize(text, locale):
    if not isinstance(text, str) or not text.strip() or len(text) > 100000:
        raise ValueError('Text must contain 1 to 100000 characters')
    # NLP pipelines mutate tokens and have shared state; serialize G2P only.
    with _lock:
        p = pipeline(locale)
        chunks = []
        for segment in re.findall(r'[^\n]+', text):
            if not segment.strip():
                continue
            _, tokens = p.g2p(segment)
            if any(not t.phonemes and any(c.isalnum() for c in t.text) for t in tokens):
                raise ValueError('G2P could not pronounce every source token')
            for _, ps, _ in p.en_tokenize(tokens):
                if not ps or len(ps) > 510:
                    raise ValueError('Invalid or oversized phoneme chunk; no silent truncation')
                chunks.append(ps)
        if not chunks or len(chunks) > 4096:
            raise ValueError('Empty or oversized G2P result')
        return chunks

def validate_stored(value, text, locale):
    if not isinstance(value, dict) or set(value) != {'inputHash', 'g2pVersion', 'phonemes'}:
        raise ValueError('Invalid stored phoneme record')
    if value['inputHash'] != hashlib.sha256(text.encode()).hexdigest():
        raise ValueError('Stored phonemes differ from source text')
    if value['g2pVersion'] != g2p_version():
        raise ValueError('Stored phonemes use a different G2P version; regenerate them')
    chunks = value['phonemes']
    if not isinstance(chunks, list) or not chunks or len(chunks) > 4096 or any(not isinstance(p, str) or not p.strip() or len(p) > 510 for p in chunks):
        raise ValueError('Invalid stored phoneme chunks')
    pipeline(locale)
    return value

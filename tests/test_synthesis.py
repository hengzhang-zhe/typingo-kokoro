import asyncio
import hashlib
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch
import pytest
from app import batch
from app.synthesis import speed_string, validate_options, override_tokens, for_voice, generate_contextual


@pytest.mark.parametrize('value', [0, 2.1, 'NaN', float('inf'), True, None])
def test_invalid_speed(value):
    with pytest.raises(ValueError):
        speed_string(value)


def test_contextual_override_preserves_other_tokens_and_utf16_offsets():
    text = '😀 I read it.'
    override = dict(start=5, end=9, text='read', ipa='red', locale='en-US')
    item = dict(text=text, synthesis=dict(mode='mixed', speed=0.850, overrides=[override]))
    options = validate_options(item, required=True)
    assert options['speed'] == '0.85'
    tokens = [types.SimpleNamespace(text=t, phonemes=p) for t, p in [('😀', ''), ('I', 'I'), ('read', 'ɹid'), ('it', 'ɪt'), ('.', '.')]]
    result = override_tokens(text, tokens, [override], 'en-US')
    assert [t.phonemes for t in result] == ['', 'I', 'ɹɛd', 'ɪt', '.']
    assert for_voice(options, 'en-GB')['mode'] == 'text'
    with pytest.raises(ValueError, match='source span'):
        validate_options(dict(item, synthesis=dict(options, overrides=[dict(override, start=4)])))
    with pytest.raises(ValueError, match='complete contextual token'):
        override_tokens(text, tokens, [dict(override, start=5, end=8, text='rea')], 'en-US')


def test_mode_requires_real_target_and_does_not_fall_back():
    with pytest.raises(ValueError, match='mode'):
        validate_options(dict(text='read', synthesis=dict(mode='ipa', speed='1.0', overrides=[])), True)
    with pytest.raises(ValueError, match='required'):
        validate_options(dict(text='read'), True)


def test_speed_runtime_changes_and_resume_keep_audio_separate(tmp_path):
    calls = []
    class Engine:
        def __init__(self): self.model_hash = 'one'
        def provenance(self, voice): return dict(model='hexgrad/Kokoro-82M', assetsSha256={'model': self.model_hash})
        async def synthesize_to_file(self, **kwargs):
            calls.append(kwargs['speed'])
            kwargs['target'].write_bytes(str(kwargs['speed']).encode())
            return dict(phonemes=['hɛlO'])
    engine = Engine()
    stub = types.ModuleType('app.engine'); stub.get_engine = lambda: engine
    def task(speed):
        identity = dict(text='Hello.', contentType='sentence', language='en', ipa=None, locale='en-US', voice='af_heart', provider='kokoro', model='hexgrad/Kokoro-82M', overrides=[], speed=speed, format='mp3')
        return dict(format='typingo-tts-export', voices=[dict(voice='af_heart', locale='en-US')], items=[dict(contentId='item', text='Hello.', voices=['af_heart'], snapshot=hashlib.sha256(b'Hello.\0\0\0').hexdigest(), identities={'af_heart': identity}, synthesis=dict(mode='text', speed=speed, overrides=[]))])
    async def run():
        jobs = []
        for speed in ['0.85', '1.0', '0.85']:
            view = await batch.import_task(json.dumps(task(speed)).encode())
            job = batch.get_job(view['id']); jobs.append(job)
            await batch._run_one(job)
            assert job.completed == 1 and job.failed == 0
        assert calls == [0.85, 1.0]
        assets = [json.loads((j.root / 'assets.jsonl').read_text()) for j in jobs]
        assert assets[0]['objectKey'] != assets[1]['objectKey']
        assert assets[0]['objectKey'] == assets[2]['objectKey']
        assert assets[2]['synthesis']['phonemes'] == ['hɛlO']
        await batch._run_one(jobs[0]); assert calls == [0.85, 1.0]
        engine.model_hash = 'two'
        await batch._run_one(jobs[0]); assert calls == [0.85, 1.0, 0.85]
        assert json.loads((jobs[0].root / 'assets.jsonl').read_text())['objectKey'] != assets[0]['objectKey']
    with patch.object(batch, 'OUTPUT_ROOT', tmp_path), patch.dict(sys.modules, {'app.engine': stub}):
        asyncio.run(run())


def test_mixed_mode_keeps_newline_boundaries_and_absolute_offsets():
    seen = []
    class Pipeline:
        def g2p(self, text):
            seen.append(text)
            return '', [types.SimpleNamespace(text=t, phonemes='default') for t in text.split()]
        def generate_from_tokens(self, tokens, **kwargs):
            yield [t.phonemes for t in tokens]
    overrides = [dict(start=9, end=13, text='read', ipa='red', locale='en-US')]
    results = list(generate_contextual(Pipeline(), 'Hello.\nI read it.', overrides, 'en-US', 'af_heart', 1.0, None))
    assert seen == ['Hello.', 'I read it.']
    assert results == [['default'], ['default', 'ɹɛd', 'default']]

import hashlib
from unittest.mock import patch
import pytest
from app.g2p import phonemize, validate_stored

@pytest.mark.parametrize('word', ['strike', 'water', 'right'])
@pytest.mark.parametrize('locale', ['en-US', 'en-GB'])
def test_real_misaki_word_conversion(word, locale):
    chunks = phonemize(word, locale)
    assert chunks and all(isinstance(p, str) and 0 < len(p) <= 510 for p in chunks)
    assert '/' not in ''.join(chunks)

def test_british_and_american_water_are_separate():
    assert phonemize('water', 'en-US') != phonemize('water', 'en-GB')

def test_paragraphs_preserve_chunk_boundaries_without_audio():
    assert phonemize('Strike right.\nWater is here.', 'en-US') == phonemize('Strike right.', 'en-US') + phonemize('Water is here.', 'en-US')

def test_stale_source_and_g2p_version_are_rejected():
    value = {'inputHash': hashlib.sha256(b'water').hexdigest(), 'g2pVersion': 'v1', 'phonemes': ['w']}
    with patch('app.g2p.g2p_version', return_value='v1'):
        assert validate_stored(value, 'water', 'en-US') == value
        with pytest.raises(ValueError, match='source text'):
            validate_stored(value, 'right', 'en-US')
    with patch('app.g2p.g2p_version', return_value='v2'):
        with pytest.raises(ValueError, match='different G2P version'):
            validate_stored(value, 'water', 'en-US')

def test_invalid_locale_and_empty_text_are_rejected():
    with pytest.raises(ValueError): phonemize('water', 'fr-FR')
    with pytest.raises(ValueError): phonemize(' ', 'en-US')


def test_http_batch_returns_each_locale_and_isolates_an_invalid_item():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api import router
    app=FastAPI(); app.include_router(router)
    with TestClient(app) as client:
        version=client.get('/v1/phonemes/version').json()['g2pVersion']
        response=client.post('/v1/phonemes',json={'items':[
            {'id':'us','text':'water','locale':'en-US'},
            {'id':'gb','text':'water','locale':'en-GB'},
            {'id':'bad','text':'water','locale':'fr-FR'}]})
        assert response.status_code==200
        result=response.json(); assert result['g2pVersion']==version
        assert result['items'][0]['phonemes'] != result['items'][1]['phonemes']
        assert result['items'][2]['error'] and not result['items'][2]['phonemes']
        assert client.post('/v1/phonemes',json={'items':[{'id':'x','text':'water','locale':'en-US'}]*2}).status_code==400


def test_saved_phonemes_drive_audio_cache_and_resume(tmp_path):
    import asyncio,json,types,sys
    from app import batch
    from current_task import current_task
    captured=[]
    class Engine:
        async def synthesize_to_file(self,**kwargs):
            captured.append(kwargs['phonemes'])
            kwargs['target'].write_bytes(b'audio')
            return dict(phonemes=kwargs['phonemes'])
    engine=types.ModuleType('app.engine'); engine.get_engine=lambda:Engine()
    stored=dict(inputHash=hashlib.sha256(b'water').hexdigest(),g2pVersion='v1',phonemes=['w'])
    task=current_task(dict(format='typingo-tts-export',voices=[dict(voice='af_heart',locale='en-US')],items=[dict(contentId='water',text='water',snapshot=hashlib.sha256(b'water\0\0\0').hexdigest(),voices=['af_heart'],phonemes={'en-US':stored})]))
    async def run():
        view=await batch.import_task(json.dumps(task).encode()); job=batch.get_job(view['id'])
        await batch._run_one(job); assert job.completed==1 and job.failed==0
        await batch._run_one(job); assert job.completed==1 and job.initial_completed==1
        asset=json.loads((job.root/'assets.jsonl').read_text().splitlines()[0])
        assert asset['synthesis']['g2pVersion']=='v1'
        assert asset['synthesis']['inputMode']=='persisted-phonemes'
        assert asset['synthesis']['phonemes']==['w']
        second=await batch.import_task(json.dumps(task).encode()); await batch._run_one(batch.get_job(second['id']))
        assert len(captured)==1  # Persistent reusable audio cache.
        with patch('app.g2p.g2p_version',return_value='v2'):
            with pytest.raises(ValueError,match='different G2P version'): await batch._run_one(job)
    with patch('app.g2p.g2p_version',return_value='v1'),patch.object(batch,'OUTPUT_ROOT',tmp_path),patch.dict(sys.modules,{'app.engine':engine}):
        asyncio.run(run())

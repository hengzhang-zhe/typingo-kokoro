import asyncio, hashlib, json, sys, types
from pathlib import Path
from unittest.mock import patch
import pytest
from app.pronunciation import ipa_to_misaki
from app import batch

@pytest.mark.parametrize('ipa,locale,expected', [('ˈrekərd','en-US','ɹˈɛkəɹd'),('rɪˈkɔːrd','en-US','ɹɪkˈɔɹd'),('liːd','en-US','lid'),('led','en-GB','lɛd'),('ˈrekɔːd','en-GB','ɹˈɛkɔːd')])
def test_explicit_phoneme_mapping(ipa,locale,expected):
    assert ipa_to_misaki(ipa,locale)==expected

@pytest.mark.parametrize('ipa,locale',[('/led/','en-US'),('l(r)ed','en-GB'),('lXed','en-US'),('led','fr-FR'),('ləʊd','en-US')])
def test_unknown_or_ambiguous_phonetics_fail(ipa,locale):
    with pytest.raises(ValueError): ipa_to_misaki(ipa,locale)

def test_two_readings_same_voice_have_independent_audio_and_resume(tmp_path):
    captured=[]
    class Engine:
        async def synthesize_to_file(self,**kwargs):
            captured.append(kwargs['phonemes'])
            kwargs['target'].write_bytes(kwargs['phonemes'].encode())
    stub=types.ModuleType('app.engine');stub.get_engine=lambda:Engine()
    items=[]
    for pid,ipa in [('noun','ˈrekərd'),('verb','rɪˈkɔːrd')]:
        snapshot=hashlib.sha256(('record\0'+pid+'\0en-US\0'+ipa).encode()).hexdigest()
        items.append(dict(contentId='word',text='record',pronunciationId=pid,locale='en-US',ipa=ipa,snapshot=snapshot,voices=['af_heart']))
    task=dict(schemaVersion='typingo-tts-export/v3',voices=[dict(voice='af_heart',locale='en-US')],items=items)
    async def run():
        view=await batch.import_task(json.dumps(task).encode());job=batch._jobs[view['id']]
        await batch._run_one(job)
        assert job.completed==2 and job.failed==0
        await batch._run_one(job)
        assert job.completed==2 and job.failed==0
        assets=[json.loads(x) for x in (job.root/'assets.jsonl').read_text().splitlines()]
        assert len({a['objectKey'] for a in assets})==2
        assert {a['pronunciationId'] for a in assets}=={'noun','verb'}
    with patch.object(batch,'OUTPUT_ROOT',tmp_path),patch.dict(sys.modules,{'app.engine':stub}):
        asyncio.run(run())
    assert len(captured)==2 and captured[0]!=captured[1]

def test_unsupported_reading_is_reported_without_discarding_supported_targets(tmp_path):
    def target(pid,ipa):
        return dict(contentId='word',text='record',pronunciationId=pid,locale='en-US',ipa=ipa,snapshot=hashlib.sha256(('record\0'+pid+'\0en-US\0'+ipa).encode()).hexdigest(),voices=['af_heart'])
    task=dict(schemaVersion='typingo-tts-export/v3',voices=[dict(voice='af_heart',locale='en-US')],items=[target('good','ˈrɛkɚd'),target('ambiguous','(for v.) rɪˈkɔrd')])
    with patch.object(batch,'OUTPUT_ROOT',tmp_path):
        view=asyncio.run(batch.import_task(json.dumps(task).encode()))
        assert view['audioCount']==1 and view['skippedCount']==1
        assert view['skipped'][0]['pronunciationId']=='ambiguous'
        job=batch.get_job(view['id'])
        assert json.loads(job.source_path.read_text())['skipped']==view['skipped']

def test_resume_rejects_asset_target_drift_even_when_file_checksum_matches(tmp_path):
    ipa='ˈrɛkɚd';snapshot=hashlib.sha256(('record\0noun\0en-US\0'+ipa).encode()).hexdigest()
    task=dict(schemaVersion='typingo-tts-export/v3',voices=[dict(voice='af_heart',locale='en-US')],items=[dict(contentId='word',text='record',pronunciationId='noun',locale='en-US',ipa=ipa,snapshot=snapshot,voices=['af_heart'])])
    with patch.object(batch,'OUTPUT_ROOT',tmp_path):
        view=asyncio.run(batch.import_task(json.dumps(task).encode()));job=batch.get_job(view['id'])
        file=job.root/'audio'/'kokoro'/'word'/snapshot/'af_heart.mp3';file.parent.mkdir(parents=True);file.write_bytes(b'audio')
        asset=dict(contentId='word',pronunciationId='noun',voice='af_heart',locale='en-US',role='pronunciation',ipa='different',snapshot=snapshot,objectKey=file.relative_to(job.root).as_posix(),sizeBytes=5,checksumSha256=hashlib.sha256(b'audio').hexdigest())
        journal=job.root/'assets.jsonl';journal.write_text(json.dumps(asset)+'\n')
        assert batch._recover_assets(job,journal)==set()
        assert journal.read_text()==''

import asyncio
import hashlib
import json
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from app import batch


class AudioIdentityTest(unittest.TestCase):
    def identity(self):
        return dict(text='Hello.', contentType='sentence', language='en', ipa=None, locale='en-US',
                    voice='af_heart', provider='kokoro', model='hexgrad/Kokoro-82M',
                    overrides=[], speed='1.0', format='mp3')

    def task(self, content_id):
        return dict(format='typingo-tts-export', outputMode='bundle',
                    voices=[dict(voice='af_heart', locale='en-US')], items=[dict(contentId=content_id,
                    text='Hello.', voices=['af_heart'], snapshot=hashlib.sha256(b'Hello.\0\0\0').hexdigest(),
                    synthesis=dict(mode='text', speed='1.0', overrides=[]), identities={'af_heart': self.identity()})])

    def test_reimported_ids_reuse_generation_and_corrupt_cache_regenerates(self):
        calls = []
        class Engine:
            concurrency = 2
            async def synthesize(self, **kwargs):
                calls.append(kwargs['text'])
                return b'audio'
        stub = types.ModuleType('app.engine'); stub.get_engine = lambda: Engine()
        async def run(cid):
            view = await batch.import_task(json.dumps(self.task(cid)).encode())
            job = batch.get_job(view['id'])
            await batch._run_one(job)
            self.assertEqual(job.failed, 0, job.error)
            self.assertEqual(job.completed, 1)
            with zipfile.ZipFile(job.zip_path) as archive:
                manifest = json.loads(archive.read('audio-manifest.json'))
                asset = manifest['assets'][0]
                self.assertEqual(asset['contentId'], cid)
                self.assertEqual(asset['identity'], self.identity())
                self.assertEqual(archive.read(asset['objectKey']), b'audio')
                return asset['objectKey']
        with tempfile.TemporaryDirectory() as directory, patch.object(batch, 'OUTPUT_ROOT', Path(directory)), patch.dict(sys.modules, {'app.engine': stub}):
            batch.open_generation()
            first = asyncio.run(run('old-id'))
            self.assertEqual(asyncio.run(run('new-id')), first)
            self.assertEqual(len(calls), 1)
            cache = Path(directory) / '.audio-cache' / (batch._identity_key(self.identity()) + '.mp3')
            cache.write_bytes(b'bad')
            asyncio.run(run('third-id'))
            self.assertEqual(len(calls), 2)
            batch._jobs.clear()

    def test_grammar_task_keeps_example_text_and_distinct_identity(self):
        data = self.task('grammar-id')
        data['items'][0]['identities']['af_heart']['contentType'] = 'grammar'
        with tempfile.TemporaryDirectory() as directory, patch.object(batch, 'OUTPUT_ROOT', Path(directory)):
            view = asyncio.run(batch.import_task(json.dumps(data).encode()))
            try:
                job = batch.get_job(view['id'])
                normalized = json.loads(job.source_path.read_text(encoding='utf-8'))
                self.assertEqual(normalized['items'][0]['text'], 'Hello.')
                self.assertEqual(normalized['items'][0]['identities']['af_heart']['contentType'], 'grammar')
                self.assertNotEqual(batch._identity_key(normalized['items'][0]['identities']['af_heart']),
                                    batch._identity_key(self.identity()))
            finally:
                batch._jobs.pop(view['id'], None)

    def test_missing_or_changed_parameters_are_rejected(self):
        data = self.task('a')
        del data['items'][0]['identities']
        with self.assertRaisesRegex(ValueError, 'synthesis identity'):
            asyncio.run(batch.import_task(json.dumps(data).encode()))
        data = self.task('a'); data['items'][0]['identities']['af_heart']['speed'] = '2.0'
        with self.assertRaisesRegex(ValueError, 'parameters'):
            asyncio.run(batch.import_task(json.dumps(data).encode()))

    def test_ids_are_excluded_but_phonemes_and_voice_change_keys(self):
        a = self.identity()
        self.assertEqual(batch._identity_key(a), '56265a5287a8a1cf4616091b025b19c8db9f7bb65fd9553920adf19441884243' )
        self.assertNotEqual(batch._identity_key(a), batch._identity_key(dict(a, voice='af_bella')))
        self.assertNotEqual(batch._identity_key(a), batch._identity_key(dict(a, ipa='həˈləʊ')))

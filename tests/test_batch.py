"""Batch contract tests use fake synthesis; no model download is required."""
import asyncio
import torch  # Keep native Torch loaded outside patch.dict module restoration.
import gzip
import hashlib
import io
import json
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from app import batch
from app import audio  # Load native NumPy/soundfile before temporarily replacing modules.


class BatchExchangeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root_patch = patch.object(batch, 'OUTPUT_ROOT', Path(self.temp.name))
        self.root_patch.start()
        batch._jobs.clear()
        batch.open_generation()

    def tearDown(self):
        batch._jobs.clear()
        self.root_patch.stop()
        self.temp.cleanup()

    def task(self, mode='bundle'):
        return {'schemaVersion': 'typingo-tts-export/v3', 'outputMode': mode,
                'voices': [{'voice': 'af_heart', 'locale': 'en-US'}, {'voice': 'bf_alice', 'locale': 'en-GB'}],
                'items': [{'contentId': 'item-1', 'text': 'Hello.', 'snapshot': hashlib.sha256(b'Hello.\0\0\0').hexdigest(), 'voices': ['af_heart', 'bf_alice']}]}

    def raw(self, data):
        return json.dumps(data).encode()

    def test_batch_workers_allow_recovery_above_current_limit(self):
        from app.adaptive import AdjustableLimiter
        class Engine:
            capacity = 1
            concurrency_ceiling = 2
            settings = types.SimpleNamespace(dynamic_concurrency=True)
            concurrency = 1
            def __init__(self):
                self.limiter = AdjustableLimiter(1)
                self.done = self.peak = 0
            async def synthesize(self, **kwargs):
                async with self.limiter:
                    self.peak = max(self.peak, self.limiter.active)
                    await asyncio.sleep(0.01)
                    self.done += 1
                    if self.done == 3:
                        self.concurrency = 2
                        await self.limiter.resize(2)
                    return b'audio'
        engine = Engine()
        stub = types.ModuleType('app.engine'); stub.get_engine = lambda: engine
        async def run():
            data = self.task()
            data['items'] = [dict(data['items'][0], contentId=f'item-{index}') for index in range(8)]
            view = await batch.import_task(self.raw(data))
            job = batch._jobs[view['id']]
            await batch._run_one(job)
            return job
        with patch.dict(sys.modules, {'app.engine': stub}):
            job = asyncio.run(run())
        self.assertEqual(job.completed, 16)
        self.assertEqual(job.failed, 0)
        self.assertEqual(engine.peak, 2)

    def archive(self, name, raw):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(name, raw)
        return buffer.getvalue()

    def test_json_zip_gzip_same_task(self):
        raw = self.raw(self.task())
        for payload in [raw, self.archive('task.json', raw), gzip.compress(raw)]:
            view = asyncio.run(batch.import_task(payload))
            self.assertEqual(view['audioCount'], 2)
            self.assertEqual(view['outputMode'], 'bundle')

    def test_v1_compatibility(self):
        data = {'schemaVersion': 'typingo-tts-export/v1', 'items': [
            {'contentId': 'item-1', 'text': 'Hello.', 'type': 'word', 'variants': [{'voice': 'af_heart', 'locale': 'en-US'}]}]}
        with self.assertRaisesRegex(ValueError, 'schemaVersion'):
            asyncio.run(batch.import_task(self.raw(data)))

    def test_reject_unknown_voice_reference(self):
        data = self.task()
        data['items'][0]['voices'] = ['am_unknown']
        with self.assertRaisesRegex(ValueError, 'Unknown voice'):
            asyncio.run(batch.import_task(self.raw(data)))

    def test_reject_invalid_locale(self):
        data = self.task()
        data['voices'][0]['locale'] = 'en-GB'
        with self.assertRaisesRegex(ValueError, 'locale'):
            asyncio.run(batch.import_task(self.raw(data)))

    def test_reject_duplicate_content(self):
        data = self.task()
        data['items'].append(data['items'][0])
        with self.assertRaisesRegex(ValueError, 'Duplicate contentId'):
            asyncio.run(batch.import_task(self.raw(data)))

    def test_reject_duplicate_voice(self):
        data = self.task()
        data['items'][0]['voices'] = ['af_heart', 'af_heart']
        with self.assertRaisesRegex(ValueError, 'unique'):
            asyncio.run(batch.import_task(self.raw(data)))

    def test_accept_10000_reject_10001(self):
        data = self.task()
        data['items'] = [{'contentId': f'item-{i}', 'text': 'Hello.', 'snapshot': hashlib.sha256(b'Hello.\0\0\0').hexdigest(), 'voices': ['af_heart']} for i in range(10000)]
        self.assertEqual(asyncio.run(batch.import_task(self.raw(data)))['itemCount'], 10000)
        data['items'].append({'contentId': 'item-overflow', 'text': 'Hello.', 'snapshot': hashlib.sha256(b'Hello.\0\0\0').hexdigest(), 'voices': ['af_heart']})
        with self.assertRaisesRegex(ValueError, '10000'):
            asyncio.run(batch.import_task(self.raw(data)))

    def test_reject_traversal(self):
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            asyncio.run(batch.import_task(self.archive('../task.json', self.raw(self.task()))))

    def test_reject_invalid_output_mode(self):
        with self.assertRaisesRegex(ValueError, 'outputMode'):
            asyncio.run(batch.import_task(self.raw(self.task('unknown'))))

    def test_reject_expansion_over_limit(self):
        with patch.object(batch, 'MAX_TASK_JSON_BYTES', 10):
            with self.assertRaisesRegex(ValueError, 'too large'):
                batch.decode_task(gzip.compress(self.raw(self.task())))

    def generate(self, mode, fail_voice=None):
        class Engine:
            async def synthesize(self, **kwargs):
                if kwargs['voice'] == fail_voice:
                    raise ValueError('simulated synthesis failure')
                return kwargs['voice'].encode()
        stub = types.ModuleType('app.engine')
        stub.get_engine = lambda: Engine()
        async def run():
            view = await batch.import_task(self.raw(self.task(mode)))
            job = batch.get_job(view['id'])
            await batch._run(job)
            return job
        with patch.dict(sys.modules, {'app.engine': stub}):
            return asyncio.run(run())

    def test_bundle_contains_manifest_and_exact_audio(self):
        job = self.generate('bundle')
        self.assertEqual(job.status, 'completed')
        with zipfile.ZipFile(job.zip_path) as archive:
            manifest = json.loads(archive.read('audio-manifest.json'))
            self.assertEqual(set(archive.namelist()), {'audio-manifest.json', *(a['objectKey'] for a in manifest['assets'])})
            self.assertEqual(len(manifest['assets']), 2)
            self.assertNotIn('storageProvider', manifest['assets'][0])
            for asset in manifest['assets']:
                import hashlib
                self.assertEqual(hashlib.sha256(archive.read(asset['objectKey'])).hexdigest(), asset['checksumSha256'])

    def test_manifest_only_result_but_audio_kept_locally(self):
        job = self.generate('manifest')
        with zipfile.ZipFile(job.zip_path) as archive:
            self.assertEqual(archive.namelist(), ['audio-manifest.json'])
            manifest = json.loads(archive.read('audio-manifest.json'))
            for asset in manifest['assets']:
                self.assertTrue((job.root / asset['objectKey']).exists())

    def test_http_downloads_preserve_output_mode_and_separate_audio(self):
        import importlib
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        stub = types.ModuleType('app.engine')
        stub.get_engine = lambda: None
        with patch.dict(sys.modules, {'app.engine': stub}):
            api = importlib.import_module('app.api')
        app = FastAPI()
        app.include_router(api.router)
        job = self.generate('manifest')
        batch._jobs.clear()
        self.assertEqual(batch.get_job(job.id).output_mode, 'manifest')
        with TestClient(app) as client:
            response = client.post('/v1/batches/import', files={'file': ('task.zip', self.archive('task.json', self.raw(self.task('manifest'))), 'application/zip')})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['outputMode'], 'manifest')
            response = client.get(f'/v1/batches/{job.id}/download')
            self.assertEqual(response.status_code, 200)
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                self.assertEqual(archive.namelist(), ['audio-manifest.json'])
            response = client.get(f'/v1/batches/{job.id}/audio/download')
            self.assertEqual(response.status_code, 200)
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                self.assertEqual(len(archive.namelist()), 2)
                self.assertTrue(all(name.startswith('audio/') for name in archive.namelist()))
            response = client.get('/v1/audio/voices/download')
            self.assertEqual(response.status_code, 200)
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                catalog = json.loads(archive.read('voice-catalog.json'))
                self.assertEqual(catalog['schemaVersion'], 'kokoro-voice-catalog/v1')
                self.assertGreater(catalog['count'], 0)

    def test_large_bundle_splits_into_independently_parseable_packages(self):
        with patch.object(batch, 'MAX_RESULT_PART_BYTES', 1):
            job = self.generate('bundle')
        self.assertEqual(len(job.zip_paths), 2)
        found = []
        for path in job.zip_paths:
            with zipfile.ZipFile(path) as archive:
                manifest = json.loads(archive.read('audio-manifest.json'))
                self.assertEqual(len(manifest['assets']), 1)
                for asset in manifest['assets']:
                    self.assertIn(asset['objectKey'], archive.namelist())
                    found.append(asset['voice'])
        self.assertEqual(found, ['af_heart', 'bf_alice'])
        batch._jobs.clear()
        self.assertEqual(len(batch.get_job(job.id).zip_paths), 2)

    def test_manifest_parts_have_no_audio_and_preserve_all_updates(self):
        with patch.object(batch, 'MAX_MANIFEST_PART_ASSETS', 1):
            job = self.generate('manifest')
        self.assertEqual(len(job.zip_paths), 2)
        voices = []
        for path in job.zip_paths:
            with zipfile.ZipFile(path) as archive:
                self.assertEqual(archive.namelist(), ['audio-manifest.json'])
                voices.extend(asset['voice'] for asset in json.loads(archive.read('audio-manifest.json'))['assets'])
        self.assertEqual(voices, ['af_heart', 'bf_alice'])

    def test_http_chunk_upload_reassembles_zip_and_recovers_missing_part(self):
        import importlib
        import uuid
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        stub = types.ModuleType('app.engine'); stub.get_engine = lambda: None
        with patch.dict(sys.modules, {'app.engine': stub}):
            api = importlib.import_module('app.api')
        app = FastAPI(); app.include_router(api.router)
        raw = self.archive('task.json', self.raw(self.task()))
        upload_id = str(uuid.uuid4())
        metadata = {'uploadId': upload_id, 'totalChunks': 2, 'totalSize': len(raw)}
        with TestClient(app) as client:
            first = client.post('/v1/batches/import/chunks', data={**metadata, 'index': 0}, files={'file': ('part', raw[:7])})
            self.assertEqual(first.status_code, 200)
            missing = client.post('/v1/batches/import/chunks/complete', json=metadata)
            self.assertEqual(missing.status_code, 400)
            changed = client.post('/v1/batches/import/chunks', data={**metadata, 'index': 1, 'totalSize': len(raw)+1}, files={'file': ('part', raw[7:])})
            self.assertEqual(changed.status_code, 400)
            second = client.post('/v1/batches/import/chunks', data={**metadata, 'index': 1}, files={'file': ('part', raw[7:])})
            self.assertEqual(second.status_code, 200)
            completed = client.post('/v1/batches/import/chunks/complete', json=metadata)
            self.assertEqual(completed.status_code, 200)
            self.assertEqual(completed.json()['audioCount'], 2)
            self.assertFalse(api.upload_root(upload_id).exists())

    def test_long_waveform_streams_without_numpy_concatenation(self):
        import importlib
        import numpy as np
        import soundfile as sf
        from app.config import Settings
        fake_kokoro = types.ModuleType('kokoro')
        fake_kokoro.KModel = object; fake_kokoro.KPipeline = object
        with patch.dict(sys.modules, {'kokoro': fake_kokoro}):
            sys.modules.pop('app.engine', None)
            engine_module = importlib.import_module('app.engine')
        engine = engine_module.KokoroServeEngine(Settings())
        engine.model = object()
        engine.pipelines['a'] = lambda *args, **kwargs: (types.SimpleNamespace(audio=np.ones(1000, dtype=np.float32)) for _ in range(50))
        target = Path(self.temp.name) / 'streamed.mp3'
        def encode(command, **kwargs):
            info = sf.info(command[command.index('-i') + 1])
            self.assertEqual(info.frames, 50000)
            Path(command[-1]).write_bytes(b'encoded')
        with patch.object(engine_module.np, 'concatenate', side_effect=AssertionError('must not concatenate')),             patch.object(engine_module.subprocess, 'run', side_effect=encode):
            asyncio.run(engine.synthesize_to_file('Long text.', 'af_heart', 1.0, target))
        self.assertEqual(target.read_bytes(), b'encoded')

    def test_partial_failures_only_include_successful_updates(self):
        job = self.generate('manifest', 'bf_alice')
        self.assertEqual(job.status, 'completed_with_errors')
        with zipfile.ZipFile(job.zip_path) as archive:
            manifest = json.loads(archive.read('audio-manifest.json'))
            self.assertEqual([asset['voice'] for asset in manifest['assets']], ['af_heart'])

    def test_parallel_workers_are_bounded_and_keep_all_assets(self):
        active = peak = 0
        class Engine:
            concurrency = 2
            async def synthesize(self, **kwargs):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.02)
                active -= 1
                return kwargs['voice'].encode()
        stub = types.ModuleType('app.engine'); stub.get_engine = lambda: Engine()
        data = self.task()
        data['items'] *= 1
        data['items'] += [{'contentId': 'item-2', 'text': 'Hello.', 'snapshot': hashlib.sha256(b'Hello.\0\0\0').hexdigest(), 'voices': ['af_heart', 'bf_alice']}]
        async def run():
            view = await batch.import_task(self.raw(data))
            job = batch.get_job(view['id'])
            await batch._run(job)
            return job
        with patch.dict(sys.modules, {'app.engine': stub}), patch.object(batch, 'get_settings', return_value=types.SimpleNamespace(max_concurrency=0)):
            job = asyncio.run(run())
        self.assertEqual(peak, 2)
        self.assertEqual(job.completed, 4)
        manifest = json.loads((job.root / 'audio-manifest.json').read_text())
        self.assertEqual(len({(a['contentId'], a['voice']) for a in manifest['assets']}), 4)

    def test_retry_preserves_success_and_regenerates_corrupt_audio(self):
        job = self.generate('manifest', 'bf_alice')
        calls = []
        class Engine:
            async def synthesize(self, **kwargs):
                calls.append(kwargs['voice'])
                return kwargs['voice'].encode()
        stub = types.ModuleType('app.engine'); stub.get_engine = lambda: Engine()
        with (job.root / 'assets.jsonl').open('a') as journal:
            journal.write('{incomplete')
        with patch.dict(sys.modules, {'app.engine': stub}):
            asyncio.run(batch._run(job))
        self.assertEqual(calls, ['bf_alice'])
        self.assertEqual(job.completed, 2)
        self.assertEqual(job.failed, 0)
        calls.clear()
        target = job.root / 'audio/kokoro/item-1' / self.task()['items'][0]['snapshot'] / 'af_heart.mp3'
        target.write_bytes(b'x' * target.stat().st_size)
        with patch.dict(sys.modules, {'app.engine': stub}):
            asyncio.run(batch._run(job))
        self.assertEqual(calls, ['af_heart'])
        self.assertEqual(job.completed, 2)

    def test_restart_marks_old_running_job_interrupted(self):
        job = self.generate('manifest')
        job.status = 'running'
        batch._save_status(job)
        batch._jobs.clear()
        recovered = batch.get_job(job.id)
        self.assertEqual(recovered.status, 'interrupted')
        self.assertEqual(recovered.completed, 2)

    def test_shutdown_cancels_active_and_queued_batches(self):
        entered = None
        active = 0
        class Engine:
            async def synthesize(self, **kwargs):
                nonlocal active
                active += 1
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    active -= 1
        stub = types.ModuleType('app.engine'); stub.get_engine = lambda: Engine()
        async def run():
            nonlocal entered
            entered = asyncio.Event()
            first = await batch.import_task(self.raw(self.task()))
            second = await batch.import_task(self.raw(self.task()))
            await batch.start_generation(first['id'])
            await entered.wait()
            await batch.start_generation(second['id'])
            await asyncio.sleep(0)
            await batch.shutdown_generation()
            self.assertEqual(active, 0)
            self.assertFalse(batch._generation_tasks)
            for view in (first, second):
                self.assertEqual(batch.get_job(view['id']).status, 'interrupted')
            with self.assertRaisesRegex(ValueError, 'shutting down'):
                await batch.start_generation(first['id'])
        with patch.dict(sys.modules, {'app.engine': stub}), patch.object(batch, '_generation_lock', asyncio.Lock()):
            asyncio.run(run())

    def test_shutdown_before_batch_task_starts_persists_interruption(self):
        async def run():
            view = await batch.import_task(self.raw(self.task()))
            await batch.start_generation(view['id'])
            await batch.shutdown_generation()
            batch._jobs.clear()
            self.assertEqual(batch.get_job(view['id']).status, 'interrupted')
        engine=types.ModuleType('app.engine');engine.get_engine=lambda:types.SimpleNamespace(concurrency=1)
        with patch.dict(sys.modules, {'app.engine':engine}):
            asyncio.run(run())


if __name__ == '__main__':
    unittest.main()

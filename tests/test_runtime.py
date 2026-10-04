import asyncio
import threading
import unittest
from unittest.mock import patch

import torch
from app.config import Settings
from app.engine import KokoroServeEngine, resolve_device


class RuntimeTest(unittest.TestCase):
    def test_device_selection_and_explicit_cuda_failure(self):
        with patch.object(torch.cuda, 'is_available', return_value=False):
            self.assertEqual(resolve_device('auto'), 'cpu')
            with self.assertRaisesRegex(RuntimeError, 'CUDA requested'):
                resolve_device('cuda')
        with patch.object(torch.cuda, 'is_available', return_value=True), patch.object(torch.cuda, 'device_count', return_value=1):
            self.assertEqual(resolve_device('auto'), 'cuda:0')
            self.assertEqual(resolve_device('cuda'), 'cuda:0')
            with self.assertRaises(ValueError):
                resolve_device('cuda:1')

    def test_slots_isolate_models_and_enable_inference_in_each_thread(self):
        engine = KokoroServeEngine(Settings(max_concurrency=2))
        engine._pool_ready = True
        for _ in range(2):
            engine._slots.put((object(), {}))
        barrier = threading.Barrier(2, timeout=5)
        def infer():
            model, _ = engine._local.runtime
            barrier.wait()
            return id(model), torch.is_inference_mode_enabled()
        async def run():
            return await asyncio.gather(engine._execute(infer), engine._execute(infer))
        results = asyncio.run(run())
        self.assertNotEqual(results[0][0], results[1][0])
        self.assertTrue(all(enabled for _, enabled in results))
        self.assertEqual(engine._slots.qsize(), 2)

    def test_cancellation_waits_for_thread_before_reusing_slot(self):
        engine = KokoroServeEngine(Settings(max_concurrency=1))
        entered = threading.Event()
        release = threading.Event()
        def infer():
            entered.set()
            release.wait(timeout=5)
        async def run():
            task = asyncio.create_task(engine._execute(infer))
            await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            await asyncio.sleep(0.02)
            self.assertFalse(task.done())
            self.assertTrue(engine.semaphore.locked())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertFalse(engine.semaphore.locked())
        asyncio.run(run())

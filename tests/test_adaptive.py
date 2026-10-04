import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.adaptive import AdjustableLimiter, AdaptivePolicy


class AdaptiveTest(unittest.TestCase):
    def snapshot(self, **changes):
        values = dict(memory_available_mb=10000, gpu_free_mb=4000, cpu_percent=20, cpu_count=20)
        values.update(changes)
        return SimpleNamespace(**values)

    def test_pressure_requires_two_samples_and_recovers_gradually(self):
        policy = AdaptivePolicy(3, 2048, 1536)
        busy = self.snapshot(gpu_free_mb=1000)
        self.assertEqual(policy.decide(3, busy, True, 20, 100), 3)
        self.assertEqual(policy.decide(3, busy, True, 20, 100), 2)
        for _ in range(2):
            self.assertEqual(policy.decide(2, self.snapshot(), True, 20, 100), 2)
        self.assertEqual(policy.decide(2, self.snapshot(), True, 20, 100), 3)

    def test_non_improving_probe_reverts_and_cools_down(self):
        policy = AdaptivePolicy(2, 2048, 1536)
        for _ in range(2):
            self.assertEqual(policy.decide(1, self.snapshot(), True, 12, 100), 1)
        self.assertEqual(policy.decide(1, self.snapshot(), True, 12, 100), 2)
        self.assertEqual(policy.decide(2, self.snapshot(), True, 12, 101), 2)
        self.assertEqual(policy.decide(2, self.snapshot(), True, 12, 101), 1)
        for _ in range(3):
            self.assertEqual(policy.decide(1, self.snapshot(), True, 12, 100), 1)

    def test_single_fast_sample_cannot_keep_an_unhelpful_expansion(self):
        policy = AdaptivePolicy(2, 2048, 0)
        for _ in range(3):
            limit = policy.decide(1, self.snapshot(), True, 12, 100)
        self.assertEqual(limit, 2)
        self.assertEqual(policy.decide(2, self.snapshot(), True, 12, 180), 2)
        self.assertIsNotNone(policy.probe)
        self.assertEqual(policy.decide(2, self.snapshot(), True, 12, 20), 1)

    def test_improving_probe_stays_and_idle_never_expands(self):
        policy = AdaptivePolicy(2, 2048, 0)
        for _ in range(5):
            self.assertEqual(policy.decide(1, self.snapshot(), False, 0, 0), 1)
        for _ in range(3):
            limit = policy.decide(1, self.snapshot(), True, 12, 100)
        self.assertEqual(limit, 2)
        self.assertEqual(policy.decide(2, self.snapshot(), True, 12, 120), 2)
        self.assertEqual(policy.decide(2, self.snapshot(), True, 12, 120), 2)

    def test_memory_and_cpu_pressure_reduce_but_never_below_one(self):
        for snapshot in [self.snapshot(cpu_percent=95), self.snapshot(memory_available_mb=1000)]:
            policy = AdaptivePolicy(2, 2048, 0)
            policy.decide(2, snapshot, True, 20, 100)
            self.assertEqual(policy.decide(2, snapshot, True, 20, 100), 1)
            for _ in range(5):
                self.assertEqual(policy.decide(1, snapshot, True, 20, 100), 1)

    def test_resize_drains_inflight_without_interrupting_and_wakes_waiters(self):
        async def run():
            limiter = AdjustableLimiter(2)
            await limiter.__aenter__(); await limiter.__aenter__()
            await limiter.resize(1)
            entered = asyncio.Event()
            async def waiter():
                async with limiter:
                    entered.set()
            task = asyncio.create_task(waiter())
            await asyncio.sleep(0)
            self.assertEqual(limiter.active, 2)
            await limiter.__aexit__()
            await asyncio.sleep(0)
            self.assertFalse(entered.is_set())
            await limiter.resize(2)
            await task
            self.assertTrue(entered.is_set())
            await limiter.__aexit__()
            self.assertEqual(limiter.active, 0)
        asyncio.run(run())

    def test_cancel_waiter_does_not_leak_capacity(self):
        async def run():
            limiter = AdjustableLimiter(1)
            await limiter.__aenter__()
            task = asyncio.create_task(limiter.__aenter__())
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(limiter.waiting, 0)
            self.assertEqual(limiter.active, 1)
            await limiter.__aexit__()
        asyncio.run(run())

    def test_expand_pool_beyond_startup_and_release_idle_models_on_revert(self):
        from app.engine import KokoroServeEngine
        from app.config import Settings
        async def run():
            engine = KokoroServeEngine(Settings(_env_file=None, max_concurrency=0))
            engine.device = 'cuda:0'
            engine.resource_plan = SimpleNamespace(reserved_memory_mb=2048, reserved_vram_mb=1536)
            engine.model = object()
            engine._slots.put((engine.model, {}))
            self.assertTrue(engine._can_expand(self.snapshot()))
            with patch.object(engine, '_make_slot', return_value=(object(), {})) as loading:
                await engine._apply_concurrency(2)
            loading.assert_called_once()
            self.assertEqual(engine.capacity, 2)
            self.assertEqual(engine.semaphore.limit, 2)
            self.assertEqual(engine._slots.qsize(), 2)
            await engine._apply_concurrency(1)
            self.assertEqual(engine.capacity, 1)
            self.assertEqual(engine._slots.qsize(), 1)
            self.assertIs(engine._slots.get()[0], engine.model)
            self.assertTrue(engine._cache_dirty)
        asyncio.run(run())

    def test_busy_slot_retires_only_after_forward_completes(self):
        import threading
        from app.engine import KokoroServeEngine
        from app.config import Settings
        async def run():
            engine = KokoroServeEngine(Settings(_env_file=None, max_concurrency=2))
            engine.model = object()
            engine._slots.put((object(), {}))
            engine._slots.put((engine.model, {}))
            engine._pool_ready = True
            entered, release = threading.Event(), threading.Event()
            def infer():
                entered.set()
                release.wait(5)
            task = asyncio.create_task(engine._execute(infer))
            await asyncio.to_thread(entered.wait, 5)
            await engine._apply_concurrency(1)
            self.assertEqual(engine.capacity, 2)
            self.assertFalse(task.done())
            release.set()
            await task
            self.assertEqual(engine.capacity, 1)
            self.assertEqual(engine._slots.qsize(), 1)
        asyncio.run(run())

    def test_growth_respects_memory_cpu_and_explicit_ceiling(self):
        from app.engine import KokoroServeEngine
        from app.config import Settings
        engine = KokoroServeEngine(Settings(_env_file=None, max_concurrency=0))
        engine.device = 'cuda:0'
        engine.resource_plan = SimpleNamespace(reserved_memory_mb=2048, reserved_vram_mb=1536)
        self.assertFalse(engine._can_expand(self.snapshot(gpu_free_mb=2800)))
        self.assertFalse(engine._can_expand(self.snapshot(memory_available_mb=2600)))
        self.assertFalse(engine._can_expand(self.snapshot(cpu_count=2)))
        engine.concurrency_ceiling = 1
        self.assertFalse(engine._can_expand(self.snapshot()))

    def test_insufficient_growth_budget_never_starts_probe(self):
        policy = AdaptivePolicy(8, 2048, 1536)
        for _ in range(5):
            self.assertEqual(policy.decide(2, self.snapshot(), True, 20, 100, can_grow=False), 2)
        self.assertIsNone(policy.probe)

    def test_cancel_pool_growth_waits_for_loading_thread(self):
        import threading
        from app.engine import KokoroServeEngine
        from app.config import Settings
        async def run():
            engine = KokoroServeEngine(Settings(_env_file=None))
            entered, release = threading.Event(), threading.Event()
            def load(*args):
                entered.set(); release.wait(5)
                return object(), {}
            with patch.object(engine, '_make_slot', side_effect=load):
                task = asyncio.create_task(engine._apply_concurrency(2))
                await asyncio.to_thread(entered.wait, 5)
                task.cancel()
                await asyncio.sleep(0.01)
                self.assertFalse(task.done())
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            self.assertEqual(engine.capacity, 1)
            self.assertEqual(engine.concurrency, 1)
        asyncio.run(run())

    def test_failed_allocation_preserves_admissions_and_marks_cache_for_release(self):
        from app.engine import KokoroServeEngine
        from app.config import Settings
        async def run():
            engine = KokoroServeEngine(Settings(_env_file=None, max_concurrency=0))
            with patch.object(engine, '_make_slot', side_effect=RuntimeError('out of memory')):
                with self.assertRaisesRegex(RuntimeError, 'out of memory'):
                    await engine._apply_concurrency(2)
            self.assertEqual(engine.capacity, 1)
            self.assertEqual(engine.concurrency, 1)
            self.assertEqual(engine.semaphore.limit, 1)
            self.assertTrue(engine._cache_dirty)
        asyncio.run(run())

    def test_monitor_changes_live_engine_and_stops(self):
        from app.engine import KokoroServeEngine
        from app.config import Settings
        async def run():
            engine = KokoroServeEngine(Settings(_env_file=None, max_concurrency=2))
            engine.resource_plan = SimpleNamespace(reserved_memory_mb=2048, reserved_vram_mb=0)
            sampled = asyncio.Event()
            calls = 0
            def snapshot(device):
                nonlocal calls
                calls += 1
                return self.snapshot(cpu_percent=95)
            original_sleep = asyncio.sleep
            async def tick(delay):
                if calls >= 2:
                    sampled.set()
                    await asyncio.Future()
                await original_sleep(0)
            with patch('app.engine.startup_snapshot', side_effect=snapshot), patch('app.engine.asyncio.sleep', side_effect=tick):
                engine.start_resource_monitor()
                await asyncio.wait_for(sampled.wait(), 5)
                self.assertEqual(engine.concurrency, 1, engine.dynamic_status)
                self.assertEqual(engine.semaphore.limit, 1)
                self.assertTrue(engine.dynamic_status['enabled'])
                await engine.stop_resource_monitor()
                self.assertIsNone(engine._monitor)
        asyncio.run(run())

    def test_monitor_failure_keeps_limit_and_disabled_mode_does_not_start(self):
        from app.engine import KokoroServeEngine
        from app.config import Settings
        async def run():
            engine = KokoroServeEngine(Settings(_env_file=None, max_concurrency=2, dynamic_concurrency=False))
            engine.start_resource_monitor()
            self.assertIsNone(engine._monitor)
            engine.settings.dynamic_concurrency = True
            engine.resource_plan = SimpleNamespace(reserved_memory_mb=2048, reserved_vram_mb=0)
            sampled = asyncio.Event()
            original_sleep = asyncio.sleep
            calls = 0
            async def tick(delay):
                nonlocal calls
                calls += 1
                if calls > 1:
                    sampled.set()
                    await asyncio.Future()
                await original_sleep(0)
            with patch('app.engine.startup_snapshot', side_effect=RuntimeError('unavailable')), patch('app.engine.asyncio.sleep', side_effect=tick):
                engine.start_resource_monitor()
                await asyncio.wait_for(sampled.wait(), 5)
                self.assertEqual(engine.concurrency, 2)
                self.assertIn('failed', engine.dynamic_status['reason'])
                await engine.stop_resource_monitor()
        asyncio.run(run())

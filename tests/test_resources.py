import unittest
from unittest.mock import patch
from pathlib import Path

from app.config import Settings
from app.resources import Snapshot, plan_resources, startup_snapshot, _cgroup_limits


class ResourcePlanTest(unittest.TestCase):
    def settings(self, **kwargs):
        return Settings(_env_file=None, **kwargs)

    def snapshot(self, **kwargs):
        values = dict(cpu_count=16, cpu_percent=10, memory_total_mb=32768,
                      memory_available_mb=24000, gpu_total_mb=6144,
                      gpu_free_mb=5120, gpu_percent=5)
        values.update(kwargs)
        return Snapshot(**values)

    def test_six_gb_gpu_keeps_headroom_and_selects_two_slots(self):
        plan = plan_resources(self.settings(), 'cuda:0', self.snapshot())
        self.assertEqual(plan.device, 'cuda:0')
        self.assertEqual(plan.concurrency, 2)
        self.assertEqual(plan.reserved_vram_mb, 1536)
        self.assertEqual(plan.reserved_memory_mb, 8192)
        self.assertLessEqual(plan.concurrency * plan.torch_threads, plan.cpu_budget)

    def test_gpu_pressure_reduces_slots_or_falls_back_to_cpu(self):
        reduced = plan_resources(self.settings(), 'cuda:0', self.snapshot(gpu_percent=45))
        self.assertEqual(reduced.concurrency, 1)
        for snapshot in [self.snapshot(gpu_percent=85), self.snapshot(gpu_free_mb=2000)]:
            plan = plan_resources(self.settings(), 'cuda:0', snapshot)
            self.assertEqual(plan.device, 'cpu')
            self.assertIn('using CPU', plan.reason)

    def test_unknown_gpu_load_is_conservative(self):
        plan = plan_resources(self.settings(), 'cuda:0', self.snapshot(gpu_percent=None))
        self.assertEqual(plan.concurrency, 1)

    def test_cpu_load_and_memory_bound_parallelism(self):
        idle = plan_resources(self.settings(), 'cpu', self.snapshot(cpu_percent=0))
        busy = plan_resources(self.settings(), 'cpu', self.snapshot(cpu_percent=60))
        self.assertLess(busy.concurrency, idle.concurrency)
        small = plan_resources(self.settings(), 'cpu', self.snapshot(memory_total_mb=4096, memory_available_mb=4000))
        self.assertEqual(small.concurrency, 1)
        with self.assertRaisesRegex(RuntimeError, 'Insufficient startup'):
            plan_resources(self.settings(), 'cpu', self.snapshot(memory_available_mb=8500))

    def test_explicit_values_override_auto_parallelism(self):
        settings = self.settings(max_concurrency=3, torch_num_threads=2)
        plan = plan_resources(settings, 'cpu', self.snapshot())
        self.assertEqual(plan.concurrency, 3)
        self.assertEqual(plan.torch_threads, 2)
        self.assertFalse(plan.automatic)

    def test_model_pool_loads_resolved_concurrency_instead_of_zero(self):
        from app import engine as engine_module
        engine = engine_module.KokoroServeEngine(self.settings())
        with patch.object(engine_module, 'resolve_device', return_value='cuda:0'), \
             patch.object(engine_module, 'startup_snapshot', return_value=self.snapshot()), \
             patch.object(engine_module, 'KModel') as model, \
             patch.object(engine_module, 'KPipeline'), \
             patch.object(engine_module.torch, 'set_num_threads') as threads, \
             patch.object(engine_module, 'lower_cpu_priority', return_value=True):
            engine.load()
        self.assertEqual(engine.concurrency, 2)
        self.assertEqual(model.call_count, 2)
        self.assertEqual(engine._slots.qsize(), 2)
        threads.assert_called_once_with(engine.resource_plan.torch_threads)
        self.assertTrue(engine.below_normal_priority)

    def test_cuda_memory_snapshot_targets_selected_device(self):
        import types
        memory = types.SimpleNamespace(total=32 * 1024**3, available=24 * 1024**3)
        with patch('app.resources.psutil.Process') as process, \
             patch('app.resources.psutil.cpu_percent', return_value=[10, 30]), \
             patch('app.resources.psutil.virtual_memory', return_value=memory), \
             patch('app.resources._cgroup_limits', side_effect=lambda *args: args), \
             patch('app.resources.torch.cuda.mem_get_info', return_value=(5 * 1024**3, 6 * 1024**3)) as info, \
             patch('app.resources._gpu_utilization', return_value=12):
            process.return_value.cpu_affinity.return_value = [0, 1]
            snapshot = startup_snapshot('cuda:1')
        info.assert_called_once_with('cuda:1')
        self.assertEqual(snapshot.cpu_percent, 20)
        self.assertEqual(snapshot.gpu_percent, 12)

    def test_container_limits_prevent_using_host_memory_and_cpu_budget(self):
        values = {'cpu.max': '200000 100000', 'memory.max': str(4 * 1024**3),
                  'memory.current': str(1024**3)}
        root = Path('/sys/fs/cgroup')
        with patch('app.resources.os.name', 'posix'), \
             patch('app.resources.Path', return_value=root), \
             patch.object(type(root), 'read_text', lambda path: values[path.name]):
            cpus, total, available = _cgroup_limits(32, 64 * 1024**3, 48 * 1024**3)
        self.assertEqual(cpus, 2)
        self.assertEqual(total, 4 * 1024**3)
        self.assertEqual(available, 3 * 1024**3)

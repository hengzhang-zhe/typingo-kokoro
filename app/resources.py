"""A single startup snapshot and fixed, conservative inference resource budget."""
from dataclasses import asdict, dataclass
import csv
import io
import math
import os
from pathlib import Path
import subprocess

import psutil
import torch

from app.config import Settings

MIB = 1024 ** 2


@dataclass(frozen=True)
class Snapshot:
    cpu_count: float
    cpu_percent: float
    memory_total_mb: int
    memory_available_mb: int
    gpu_total_mb: int = 0
    gpu_free_mb: int = 0
    gpu_percent: float | None = None


@dataclass(frozen=True)
class ResourcePlan:
    device: str
    concurrency: int
    torch_threads: int
    cpu_budget: int
    reserved_memory_mb: int
    reserved_vram_mb: int
    automatic: bool
    reason: str
    snapshot: Snapshot

    def view(self):
        return asdict(self)


def _cgroup_limits(cpu_count, total, available):
    if os.name == 'nt':
        return cpu_count, total, available
    root = Path('/sys/fs/cgroup')
    try:
        quota, period = (root / 'cpu.max').read_text().split()
        if quota != 'max':
            cpu_count = min(cpu_count, int(quota) / int(period))
    except (OSError, ValueError):
        pass
    try:
        limit = (root / 'memory.max').read_text().strip()
        if limit != 'max':
            limit = int(limit)
            used = int((root / 'memory.current').read_text())
            total = min(total, limit)
            available = min(available, max(0, limit - used))
    except (OSError, ValueError):
        pass
    return cpu_count, total, available


def _gpu_utilization(device):
    # Match UUID rather than physical index: CUDA_VISIBLE_DEVICES may remap it.
    uuid = str(getattr(torch.cuda.get_device_properties(device), 'uuid', '')).removeprefix('GPU-').lower()
    if not uuid:
        return None
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=uuid,utilization.gpu',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True,
                                timeout=3, check=True)
        for row in csv.reader(io.StringIO(result.stdout)):
            if len(row) == 2 and row[0].strip().removeprefix('GPU-').lower() == uuid:
                return float(row[1].strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return None


def startup_snapshot(device: str) -> Snapshot:
    process = psutil.Process()
    try:
        affinity = process.cpu_affinity()
    except (AttributeError, psutil.Error):
        affinity = list(range(psutil.cpu_count() or 1))
    loads = psutil.cpu_percent(interval=1.0, percpu=True)
    cpu_load = sum(loads[i] for i in affinity if i < len(loads)) / max(1, len(affinity))
    memory = psutil.virtual_memory()
    cpus, total, available = _cgroup_limits(len(affinity), memory.total, memory.available)
    free_gpu = total_gpu = 0
    gpu_load = None
    if device.startswith('cuda'):
        free_gpu, total_gpu = torch.cuda.mem_get_info(device)
        gpu_load = _gpu_utilization(device)
    return Snapshot(cpus, cpu_load, total // MIB, available // MIB,
                    total_gpu // MIB, free_gpu // MIB, gpu_load)


def plan_resources(settings: Settings, device: str, snapshot: Snapshot) -> ResourcePlan:
    reserve = settings.resource_reserve_fraction
    cpu_budget = max(1, math.floor(snapshot.cpu_count * max(0, 1 - snapshot.cpu_percent / 100 - reserve)))
    ram_reserve = max(settings.reserve_memory_mb, math.ceil(snapshot.memory_total_mb * reserve))
    vram_reserve = max(settings.reserve_vram_mb, math.ceil(snapshot.gpu_total_mb * reserve)) if device.startswith('cuda') else 0
    ram_budget = max(0, snapshot.memory_available_mb - ram_reserve)
    vram_budget = max(0, snapshot.gpu_free_mb - vram_reserve)
    reason = 'startup free resources minus host reserve'
    if device.startswith('cuda') and settings.device == 'auto' and (
            vram_budget < 1536 or (snapshot.gpu_percent is not None and snapshot.gpu_percent >= 100 * (1 - reserve))):
        device = 'cpu'
        vram_reserve = 0
        reason = 'GPU busy or insufficient free VRAM; using CPU'
    # Includes independent model/NLP caches and an allowance for waveform
    # activations. These are planning estimates, not an OOM-proof upper bound.
    ram_slots = ram_budget // (768 if device.startswith('cuda') else 1536)
    if device.startswith('cuda'):
        gpu_slots = vram_budget // 1536
        load_slots = 1 if snapshot.gpu_percent is None else max(1, math.floor(
            (100 * (1 - reserve) - snapshot.gpu_percent) / 25))
        limit = min(8, cpu_budget, ram_slots, gpu_slots, load_slots)
    else:
        limit = min(8, max(1, cpu_budget // 2), ram_slots)
    if limit < 1:
        raise RuntimeError('Insufficient startup memory/VRAM after reserving host resources. '
                           'Close other workloads or adjust resource reserve settings.')
    concurrency = settings.max_concurrency or limit
    threads = settings.torch_num_threads or max(1, min(4, cpu_budget // concurrency))
    return ResourcePlan(device, concurrency, threads, cpu_budget, ram_reserve, vram_reserve,
                        settings.max_concurrency == 0, reason, snapshot)


def lower_cpu_priority() -> bool:
    try:
        process = psutil.Process()
        if os.name == 'nt':
            process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            process.nice(max(5, process.nice()))
        return True
    except (psutil.Error, OSError):
        return False

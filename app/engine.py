from __future__ import annotations

import asyncio
import subprocess
import tempfile
import os
import queue
import threading
import logging
import time
import math
from pathlib import Path
import soundfile as sf
from functools import lru_cache

import numpy as np
import torch
from kokoro import KModel, KPipeline

from app.audio import encode_audio
from app.config import Settings, get_settings
from app.voices import resolve_voice
from app.resources import plan_resources, startup_snapshot, lower_cpu_priority
from app.adaptive import AdjustableLimiter, AdaptivePolicy

class KokoroServeEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model: KModel | None = None
        self.pipelines: dict[str, KPipeline] = {}
        self.concurrency = settings.max_concurrency or 1
        self.semaphore = AdjustableLimiter(self.concurrency)
        self.capacity = self.concurrency
        self.concurrency_ceiling = settings.max_concurrency or 8
        self._pool_lock = threading.Lock()
        self._cache_dirty = False
        self._monitor = None
        self._completed = self._work = 0
        self.dynamic_status = {'enabled': False}
        self.resource_plan = None
        self.below_normal_priority = False
        self.device = settings.device
        self._slots = queue.Queue()
        self._local = threading.local()
        self._pool_ready = False
        self._stopping = threading.Event()

    def load(self) -> None:
        self._stopping.clear()
        if self.model is not None:
            return

        self.device = resolve_device(self.settings.device)
        self.resource_plan = plan_resources(self.settings, self.device, startup_snapshot(self.device))
        self.device = self.resource_plan.device
        self.concurrency = self.resource_plan.concurrency
        self.capacity = self.concurrency
        self.semaphore = AdjustableLimiter(self.concurrency)
        torch.set_num_threads(self.resource_plan.torch_threads)
        if self.resource_plan.automatic:
            self.below_normal_priority = lower_cpu_priority()
        logging.getLogger('uvicorn.error').info('Startup resource plan: %s', self.resource_plan.view())
        # Legacy weight_norm hooks and LSTM flattening mutate model state.
        # Each slot owns its model and language/voice caches; never share a
        # model between simultaneous forwards, including HTTP requests.
        slots = []
        for _ in range(self.concurrency):
            slots.append(self._make_slot())
        self.model, self.pipelines = slots[0]
        for slot in slots:
            self._slots.put(slot)
        self._pool_ready = True

    def _make_slot(self, warm=False):
        self._check_running()
        model = KModel(repo_id=self.settings.model_repo).to(self.device).eval()
        pipelines = {lang: KPipeline(lang_code=lang, repo_id=self.settings.model_repo,
                                    model=model, device=self.device) for lang in ('a', 'b')}
        if warm:
            # Exclude first-use NLP/CUDA setup from the throughput probe.
            with torch.inference_mode():
                for lang, voice in [('a', 'af_heart'), ('b', 'bf_alice')]:
                    for _ in pipelines[lang]('Ready.', voice=voice, model=model):
                        self._check_running()
        return model, pipelines

    def _trim_idle_slots(self):
        # Keep the primary slot referenced by the public model/pipelines fields.
        # Busy slots retire on return; never remove a model during inference.
        with self._pool_lock:
            kept = []
            while not self._slots.empty():
                try:
                    slot = self._slots.get_nowait()
                except queue.Empty:
                    break
                if self.capacity > self.concurrency and slot[0] is not self.model:
                    self.capacity -= 1
                    self._cache_dirty = True
                else:
                    kept.append(slot)
            for slot in kept:
                self._slots.put(slot)

    def _can_expand(self, snapshot):
        if self.concurrency >= self.concurrency_ceiling:
            return False
        cpu_budget = max(1, math.floor(getattr(snapshot, 'cpu_count', 1) * max(
            0, 1 - snapshot.cpu_percent / 100 - self.settings.resource_reserve_fraction)))
        if self.concurrency + 1 > (cpu_budget if self.device.startswith('cuda') else max(1, cpu_budget // 2)):
            return False
        needs_model = self.concurrency >= self.capacity
        ram_cost = (768 if self.device.startswith('cuda') else 1536) if needs_model else 0
        if snapshot.memory_available_mb < self.resource_plan.reserved_memory_mb + ram_cost:
            return False
        if self.device.startswith('cuda'):
            # Allow for the extra slot's model, NLP caches and activations.
            return snapshot.gpu_free_mb >= self.resource_plan.reserved_vram_mb + 1536
        return True

    async def _apply_concurrency(self, limit):
        if not 1 <= limit <= self.concurrency_ceiling or limit > self.capacity + 1:
            raise ValueError('Concurrency must be within the ceiling and grow one slot at a time')
        if limit > self.capacity:
            # Model loading is native threaded work too: on shutdown wait for
            # it, then discard it instead of leaving an orphan loading thread.
            loading = asyncio.create_task(asyncio.to_thread(self._make_slot, True))
            try:
                slot = await asyncio.shield(loading)
            except asyncio.CancelledError:
                try:
                    await loading
                except Exception:
                    pass
                raise
            except Exception:
                self._cache_dirty = True
                raise
            self._check_running()
            with self._pool_lock:
                self.concurrency = limit
                self._slots.put(slot)
                self.capacity += 1
        # Set the target before shrinking so returning threads can retire.
        self.concurrency = limit
        await self.semaphore.resize(limit)
        self._trim_idle_slots()

    def _release_cuda_cache(self):
        with self._pool_lock:
            dirty, self._cache_dirty = self._cache_dirty, False
        if dirty and self.device.startswith('cuda'):
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

    def _in_slot(self, function, *args):
        slot = self._slots.get() if self._pool_ready else None
        try:
            self._check_running()
            self._local.runtime = slot or (self.model, self.pipelines)
            with torch.inference_mode():
                return function(*args)
        finally:
            self._local.runtime = None
            if slot is not None:
                with self._pool_lock:
                    if self.capacity > self.concurrency and slot[0] is not self.model:
                        self.capacity -= 1
                        self._cache_dirty = True
                    else:
                        self._slots.put(slot)

    async def _execute(self, function, *args):
        async with self.semaphore:
            task = asyncio.create_task(asyncio.to_thread(self._in_slot, function, *args))
            try:
                result = await asyncio.shield(task)
                self._completed += 1
                self._work += len(args[0]) if args and isinstance(args[0], str) else 1
                return result
            except asyncio.CancelledError:
                # A Python thread cannot be cancelled. Keep its capacity reserved
                # until inference/encoding really ends.
                try:
                    await task
                except Exception:
                    pass
                raise

    def start_resource_monitor(self):
        if not self.settings.dynamic_concurrency or self._monitor is not None:
            return
        self.dynamic_status = {'enabled': True, 'capacity': self.capacity, 'ceiling': self.concurrency_ceiling,
                               'reason': 'Waiting for resource samples'}
        self._monitor = asyncio.create_task(self._monitor_resources(), name='kokoro-resources')

    async def stop_resource_monitor(self):
        if self._monitor is not None:
            self._monitor.cancel()
            try:
                await self._monitor
            except asyncio.CancelledError:
                pass
            self._monitor = None

    async def _monitor_resources(self):
        policy = AdaptivePolicy(self.concurrency_ceiling, self.resource_plan.reserved_memory_mb,
                                self.resource_plan.reserved_vram_mb)
        previous_time = time.monotonic()
        previous_done, previous_work = self._completed, self._work
        while True:
            await asyncio.sleep(self.settings.resource_check_seconds)
            try:
                await asyncio.to_thread(self._release_cuda_cache)
                snapshot = await asyncio.to_thread(startup_snapshot, self.device)
                now = time.monotonic()
                completed = self._completed - previous_done
                rate = (self._work - previous_work) / max(1e-6, now - previous_time)
                previous_time, previous_done, previous_work = now, self._completed, self._work
                new_limit = policy.decide(self.concurrency, snapshot,
                                          self.semaphore.waiting > 0, completed, rate,
                                          can_grow=self._can_expand(snapshot))
                if new_limit != self.concurrency:
                    try:
                        await self._apply_concurrency(new_limit)
                    except Exception:
                        policy.probe = None
                        policy.cooldown = 60
                        raise
                    # Exclude allocation/warm-up time from the next throughput
                    # interval; completed work is sampled again at this boundary.
                    previous_time, previous_done, previous_work = time.monotonic(), self._completed, self._work
                    logging.getLogger('uvicorn.error').info('Dynamic concurrency: %s (%s)', new_limit, policy.reason)
                self.dynamic_status = {'enabled': True, 'capacity': self.capacity, 'ceiling': self.concurrency_ceiling,
                                       'active': self.semaphore.active, 'waiting': self.semaphore.waiting,
                                       'reason': policy.reason, 'snapshot': snapshot.__dict__,
                                       'charactersPerSecond': round(rate, 2)}
            except Exception as exc:
                # Monitoring failure must not terminate or expand inference.
                self.dynamic_status = {'enabled': True, 'capacity': self.capacity,
                                       'reason': 'Resource sampling failed', 'error': str(exc)[:256]}

    def request_shutdown(self) -> None:
        self._stopping.set()

    def _check_running(self) -> None:
        if self._stopping.is_set():
            raise RuntimeError("Kokoro service is shutting down")

    async def synthesize(
        self,
        text: str,
        voice: str,
        speed: float,
        response_format: str,
    ) -> bytes:
        if self.model is None:
            raise RuntimeError("Kokoro engine has not been initialized")

        route = resolve_voice(voice)

        return await self._execute(
                self._synthesize_sync,
                text,
                voice,
                speed,
                response_format,
                route.lang_code,
            )

    async def synthesize_to_file(self, text: str, voice: str, speed: float, target: Path, phonemes: str | None = None) -> None:
        if self.model is None:
            raise RuntimeError("Kokoro engine has not been initialized")
        route = resolve_voice(voice)
        await self._execute(self._synthesize_file_sync, text, voice, speed, route.lang_code, target, phonemes)

    def _synthesize_file_sync(self, text: str, voice: str, speed: float, lang_code: str, target: Path, phonemes: str | None = None) -> None:
        model, pipelines = getattr(self._local, "runtime", None) or (self.model, self.pipelines)
        # Long passages append pipeline chunks to disk instead of concatenating
        # every waveform into one NumPy allocation.
        with tempfile.TemporaryDirectory(prefix="typingo-wave-") as temporary:
            waveform = Path(temporary) / "wave.wav"
            frames = 0
            with sf.SoundFile(waveform, mode="w", samplerate=self.settings.sample_rate,
                              channels=1, subtype="PCM_16", format="WAV") as output:
                results = pipelines[lang_code].generate_from_tokens(phonemes, voice=voice, speed=speed, model=model) if phonemes is not None else pipelines[lang_code](text, voice=voice, speed=speed, model=model)
                for result in results:
                    self._check_running()
                    if result.audio is not None:
                        chunk = np.asarray(result.audio, dtype=np.float32)
                        output.write(chunk)
                        frames += len(chunk)
                        if frames * 2 > 2 * 1024 * 1024 * 1024:
                            raise ValueError("Single audio waveform exceeds 2GB; split this passage task")
            if not frames:
                raise RuntimeError("Kokoro returned no audio")
            self._check_running()
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(waveform),
                            "-threads", "1", "-codec:a", "libmp3lame", "-q:a", str(self.settings.mp3_quality), str(target)], check=True)

    def _synthesize_sync(
        self,
        text: str,
        voice: str,
        speed: float,
        response_format: str,
        lang_code: str,
    ) -> bytes:
        model, pipelines = getattr(self._local, "runtime", None) or (self.model, self.pipelines)
        pipeline = pipelines[lang_code]
        chunks: list[np.ndarray] = []

        for result in pipeline(
            text,
            voice=voice,
            speed=speed,
            model=model,
        ):
            self._check_running()
            audio = result.audio
            if audio is not None:
                chunks.append(np.asarray(audio, dtype=np.float32))

        if not chunks:
            raise RuntimeError("Kokoro returned no audio")

        waveform = np.concatenate(chunks)
        self._check_running()

        return encode_audio(
            waveform,
            self.settings.sample_rate,
            response_format,
            mp3_quality=self.settings.mp3_quality,
        )

def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, cuda or cuda:N")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable. Install CUDA-enabled PyTorch and check the NVIDIA driver.")
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} does not exist")
        return f"cuda:{index}"
    return "cpu"

@lru_cache
def get_engine() -> KokoroServeEngine:
    return KokoroServeEngine(get_settings())

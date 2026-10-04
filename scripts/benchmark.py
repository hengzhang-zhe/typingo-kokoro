"""Run from the project root: python -m scripts.benchmark --device auto --count 12."""
import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path

import torch
from app.config import Settings
from app.engine import KokoroServeEngine
from app.resources import startup_snapshot


async def benchmark(args):
    engine = KokoroServeEngine(Settings(device=args.device, max_concurrency=args.concurrency))
    engine.load()
    concurrency_limit = engine.concurrency
    with tempfile.TemporaryDirectory(prefix="kokoro-benchmark-") as directory:
        root = Path(directory)
        async def generate(index):
            await engine.synthesize_to_file(args.text, 'af_heart', 1.0, root / f'{index}.mp3')
        await asyncio.gather(*(generate(i) for i in range(concurrency_limit)))
        results = []
        levels = range(1, args.sweep_max + 1) if args.sweep_max else sorted({1, concurrency_limit})
        for concurrency in levels:
            if args.sweep_max:
                if concurrency > engine.concurrency:
                    snapshot = await asyncio.to_thread(startup_snapshot, engine.device)
                    if not engine._can_expand(snapshot):
                        results.append({'concurrency': concurrency, 'skipped': 'Insufficient reserved resource budget'})
                        break
                await engine._apply_concurrency(concurrency)
                await asyncio.gather(*(generate(i) for i in range(concurrency)))
            indices = iter(range(args.count))
            async def worker():
                for index in indices:
                    await generate(index)
            start = time.perf_counter()
            await asyncio.gather(*(worker() for _ in range(concurrency)))
            seconds = time.perf_counter() - start
            results.append({'concurrency': concurrency, 'seconds': round(seconds, 3),
                            'audioPerSecond': round(args.count / seconds, 3)})
    print(json.dumps({'device': engine.device, 'torch': torch.__version__,
                      'cpuThreads': torch.get_num_threads(), 'count': args.count,
                      'resourcePlan': engine.resource_plan.view(),
                      'gpu': torch.cuda.get_device_name() if engine.device.startswith('cuda') else None,
                      'results': results}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='auto')
    parser.add_argument('--concurrency', type=int, default=0)
    parser.add_argument('--count', type=int, default=12)
    parser.add_argument('--sweep-max', type=int, default=0,
                        help='Compare warmed concurrency levels 1..N using guarded on-demand pool expansion')
    parser.add_argument('--text', default='Learning English can be easy and enjoyable.')
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')
    if not 0 <= args.sweep_max <= 8:
        parser.error('--sweep-max must be 0..8')
    asyncio.run(benchmark(args))

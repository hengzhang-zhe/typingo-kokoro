# Architecture

## Scope

Typingo Kokoro is an HTTP service wrapper around the official Kokoro inference library and Kokoro-82M model.

It deliberately does not fork a third-party Kokoro API server.

## Upstream boundary

```text
typingo-kokoro
   ↓ Python dependency
hexgrad/kokoro
   ↓ Hugging Face model download
hexgrad/Kokoro-82M
```

The official `KModel` implementation downloads `config.json` and the model weights from Hugging Face when needed.

## Runtime model

Kokoro's official model implementation documents that one `KModel` instance can be reused across multiple `KPipeline` instances to avoid redundant memory allocation.

Each inference slot therefore uses:

```text
one KModel
├─ KPipeline(lang_code="a") → American English
└─ KPipeline(lang_code="b") → British English
```

Both pipelines within a slot share one model. Different slots have independent models and pipeline caches: upstream legacy weight normalization hooks and LSTM flattening mutate state, so simultaneous forwards must not share a model. The pool is loaded at startup and is shared by batch and HTTP synthesis through a semaphore. Inference mode is enabled inside each worker thread. Cancellation waits for the native inference thread before returning its slot.

## Process model

Run one Uvicorn worker per container.

Multiple Uvicorn workers duplicate model pools and do not share in-memory batch ownership. Keep one process and tune the bounded slot pool. Separate containers require separate batch storage and explicit routing.

## Concurrency

Inference is protected by an application-level semaphore. The default is:

```text
KOKORO_SERVE_MAX_CONCURRENCY=0
```

Batch generation uses a fixed number of asynchronous workers that dispatch inference and encoding to threads. It never creates one asyncio task per audio. CPU intra-op threads are limited separately to prevent oversubscription. `device=auto` selects CUDA when supported by the installed PyTorch wheel; explicit CUDA fails on unavailable hardware.

Tune concurrency after measuring CPU/GPU utilization, RAM/VRAM and latency with `python -m scripts.benchmark`. The GPU Compose override builds a CUDA wheel and reserves one NVIDIA device.

Completed assets are flushed to an append-only journal. Resume verifies file size and SHA-256 before skipping successful audio; corrupted files and partial journal tails are regenerated. A restarted running job becomes interrupted and can be resumed in Studio. Progress snapshots are atomically replaced and rate-limited to once per second. MP3 files are stored without redundant ZIP compression.

## Storage

Hugging Face runtime assets are persisted under:

```text
./data/huggingface
```

Local Python resolves this path from the project package before importing Kokoro. Docker Compose bind-mounts the same directory at `/data/huggingface`. `HF_HUB_DISABLE_SYMLINKS=1` keeps cache entries as regular files so both Windows and the Linux container can read them. The cache is ignored by Git and is populated on first model or voice use.

Generated test files are stored under:

```text
./output
```

Neither directory should be committed to Git.

## Production integration

Typingo Kokoro should stay stateless from the product perspective.

Authentication, subscriptions, quotas, Redis metadata and permanent audio caching belong upstream in the consuming application or media service.

A recommended permanent cache identity is:

```text
SHA256(
    provider
  | model_revision
  | locale
  | voice
  | speed
  | normalized_text
)
```

For local development the Compose file binds to `127.0.0.1`. In production expose the service only on a private network.


## Server-owned generation lifetime

The lifespan tracks every background batch task and cancels/awaits them on
shutdown. A thread stop event prevents processing additional waveform chunks.
Blocking recovery and packaging operations are also awaited and observe the
batch stop event. Pending batches are saved as interrupted even if cancelled
before their coroutine starts.

Windows uses an unnamed kill-on-close job, containing the server and its native
children. Its handle is non-inheritable and remains open until the server exits.
For a multiprocessing worker (Uvicorn reload), a daemon watcher waits on the
specific parent process sentinel; parent exit terminates the worker's job.
There is no separately launched generation service in application code.

Startup resource planning uses a one-second snapshot, deducts current CPU load and host reserves, and prepares initial model slots and fixed intra-op threads. Batch and HTTP generation share an adjustable admission limiter. A lifecycle-owned sampler reduces concurrency under sustained CPU/memory pressure and probes gradual expansion, including beyond the startup pool, subject to current CPU/RAM/VRAM budgets and the ceiling (8 in automatic mode). New independent models load and warm off the event loop. Non-improving throughput probes revert with a cooldown. Idle surplus models retire on shrink; busy surplus models retire when inference completes, preserving the primary slot. Device selection stays fixed. GPU load is matched by UUID; auto device can choose CPU at startup if the GPU is busy or lacks reserved VRAM. See the README for thresholds, limitations and reserve configuration.

Portable cached wheels, recoverable installed pure Python dependencies and a reconstructed spaCy model-data wheel live in `data/wheels/common`. Docker reads them through a build bind mount; local install helpers use the same find-links directory. Model compatibility metadata is included in the joint dependency resolution. Windows native binaries are never copied into Linux site-packages. Docker pip/apt use persistent BuildKit caches, and the PyTorch layer is independent of application requirements. The Docker build helper also exports container-installed portable dependencies and language data back to the shared wheelhouse without network access.

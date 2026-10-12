# Typingo Kokoro

A self-hosted HTTP service built directly on the official Kokoro inference library and model.

## Upstream

- Inference library: `hexgrad/kokoro`
- Model repository: `hexgrad/Kokoro-82M`

Typingo Kokoro does not embed or fork third-party FastAPI/WebUI wrappers. It provides its own API, Docker packaging, runtime configuration, tests, and audio encoding around the official Kokoro interfaces.

## Features

- OpenAI-style `POST /v1/audio/speech`
- American English and British English voice routing
- MP3 / WAV / FLAC / Opus output
- Bounded parallel generation with isolated model/pipeline slots
- Automatic CUDA selection, CPU fallback, and resumable batch jobs
- CPU-first Docker deployment
- Persistent Hugging Face cache
- Health, model and voice discovery endpoints
- Windows-friendly local deployment
- Unified default port: `9000`

## Quick start

Recommended Windows location:

```text
D:\Zhe\Code\typingo-kokoro
```

```powershell
git clone https://github.com/hengzhang-zhe/typingo-kokoro.git
cd typingo-kokoro
Copy-Item .env.example .env
docker compose build
docker compose up -d
docker compose logs -f
```

Docker Compose builds the local image as `typing-kokoro:latest` and runs the `typingo-kokoro` service.

OpenAPI:

```text
http://localhost:9000/docs
```

Health:

```text
http://localhost:9000/health
```

## Local Python development

Open this project in PyCharm with a dedicated Python environment and install `requirements.txt`. Run the `uvicorn` module with `app.main:app --host 127.0.0.1 --port 9000 --reload --reload-dir app` from the project directory. The app sets `HF_HOME` to `./data/huggingface` by default, so model downloads stay inside the project without a machine-specific PyCharm setting.

Docker Compose mounts the same `./data/huggingface` directory into the container. Both runtimes use regular cache files rather than symlinks so Windows and Linux can read the same downloads. Restart the local process after changing the cache configuration. Only one runtime can bind host port `9000` at a time; change one port when running both services.

## Generate speech

```http
POST /v1/audio/speech
Content-Type: application/json

{
  "model": "kokoro",
  "input": "Learning English can be easy and enjoyable.",
  "voice": "af_heart",
  "response_format": "mp3",
  "speed": 1.0
}
```

Voice prefixes enabled by this service:

| Prefix | Locale | Description |
|---|---|---|
| `af_*` | `en-US` | American English female |
| `am_*` | `en-US` | American English male |
| `bf_*` | `en-GB` | British English female |
| `bm_*` | `en-GB` | British English male |

## Architecture

```text
HTTP API :9000
   ↓
KokoroServeEngine
   ↓
inference slots (planned at startup), each with a KModel
   ├─ KPipeline(lang_code="a")  American English
   └─ KPipeline(lang_code="b")  British English
   ↓
hexgrad/Kokoro-82M
```

Each inference slot reuses its own model across American and British English pipelines. Concurrent slots have independent models because upstream weight normalization hooks and LSTM preparation mutate state during inference.

See `docs/ARCHITECTURE.md` and `THIRD_PARTY_NOTICES.md`.


## Batch Audio Studio

Open:

```text
http://localhost:9000/
```

The web UI accepts ZIP, JSON and GZIP tasks exported from Typingo Admin, using the canonical v3 pronunciation-target contract.

Workflow:

```text
Typingo Audio Management
  -> select learning items
  -> select Kokoro voices
  -> choose bundle or manifest-only output
  -> export typingo-tts-export.zip
  -> upload to Kokoro Batch Audio Studio
  -> generate
  -> download ZIP
```

The default bundle ZIP contains:

```text
audio/
  kokoro/
    <contentId>/
      <voice>.mp3
audio-manifest.json
```

`audio-manifest.json` retains the Typingo `contentId`, locale and Kokoro voice for every generated file, so Typingo can associate imported audio with the original learning item.

The batch processor generates only the voices explicitly requested by each item. Manifest-only output contains only `audio-manifest.json`; generated audio and reports stay in the local batch directory. The command-line script uses the same batch API and packaging rules as the web UI.

### Typingo 音频交换包

任务支持 ZIP（一个 JSON）、JSON 与 GZIP，仅接受 `typingo-tts-export/v3`。顶层 `voices` 声明 `voice`/`locale`，每个 item 包含 `contentId`、`text`、`voices` 和内容快照 `snapshot`；单词读音另含 `pronunciationId`、`locale`、`ipa`。最多 10000 个内容项；只生成任务明确指定的音色。

`outputMode=bundle`（默认）结果 ZIP 包含 `audio-manifest.json` 和 `audio/`，直接回导 Typingo 即可上传并关联。`outputMode=manifest` 的结果 ZIP 只包含本次成功生成的更新清单；音频留在本地批次目录，可手动放入对象存储或单独下载音频 ZIP 再导入。报告留在本地，不进入交换包。声音目录也提供 ZIP 下载。部分失败的结果只回导成功项，重试前应重新从 Typingo 导出缺失任务。

### Large batches

Task files and their expanded JSON are limited to 32MB, with up to 10000 distinct content items and 100000 reading targets; reduce the Typingo export count when text exceeds that bound. Studio supports 4MB task-upload chunks with a 256KB fallback after HTTP 413. Generation runs one batch at a time, with bounded concurrent audio workers inside each batch. Long waveforms are appended to a temporary WAV rather than concatenated in memory; result assets and failures are staged as JSONL. Results split at approximately 256MB of audio or 5000 manifest entries. Every result part is an independent ZIP with its own manifest. Download and import all parts; never concatenate their ZIP bytes. Separate audio downloads also accept `?part=N`. The CLI downloads all result parts sequentially. A single generated audio must fit Typingo's 100MB object limit; the temporary WAV has a 2GB safety bound.


## Parallel generation and GPU

`KOKORO_SERVE_DEVICE=auto` selects CUDA when its startup load and free VRAM permit, and otherwise CPU.
Explicit `cuda` or `cuda:N` fails clearly if CUDA is unavailable. `/health` and
Studio report the actual device and current concurrency. Keep one Uvicorn
process; extra Uvicorn workers duplicate model pools and do not share batch state.

- `KOKORO_SERVE_MAX_CONCURRENCY=0`: calculate initial concurrency at startup and allow dynamic expansion up to 8; 1–8 explicitly sets initial concurrency and its upper bound. Dynamic admissions may use fewer slots.
  Each slot has a separate model and language/voice caches. Increasing this
  consumes more RAM/VRAM; benchmark before increasing it, especially for passages.
- `KOKORO_SERVE_TORCH_NUM_THREADS=0`: calculate process-level CPU intra-op threads from the startup CPU budget and selected concurrency, capped at 4; a positive value overrides it. It remains fixed while admissions change.
- `KOKORO_SERVE_MP3_QUALITY=0`: highest LAME VBR quality (0–9, lower is better).
  The previous setting was 3; use 3 for smaller output files.

Native 24kHz, float32 model inference is preserved. CUDA accelerates inference;
it does not make pronunciation or the model itself more accurate. MP3 quality
changes encoding fidelity and file size, not linguistic quality. WAV/FLAC are
available through the speech API for lossless delivery.

For local NVIDIA GPU support, activate the project's Conda environment and run:

```powershell
python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Set `KOKORO_SERVE_DEVICE=auto` in `.env`, then restart the IDE run configuration.
The PyTorch wheel includes the CUDA runtime; a compatible NVIDIA driver is required.
See [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

Docker defaults to a CPU wheel. To build CUDA support and reserve a GPU, use the
GPU override with NVIDIA Container Toolkit configured on the Docker host:

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Studio reports audio/second and an estimated remaining time based on the current
run. Restarted jobs appear as interrupted; click “继续生成未完成音频” to resume.
Successful audio is verified by size and SHA-256 and reused; missing, corrupt,
and failed audio is regenerated. Completed files retain their original encoding
quality when resumed. Journal tails from an interrupted write are discarded.
The task and its output directory must remain in `output/batches` to resume.

To compare serial and concurrent throughput on your own text:

```powershell
python -m scripts.benchmark --device auto --concurrency 2 --count 12
```


## Stopping generation with the server

Batch workers belong to the FastAPI lifespan inside the Uvicorn server process.
On normal shutdown, the service stops accepting batch starts, signals inference
threads to stop at the next chunk boundary, cancels active and queued batches,
waits for current native work to finish, and saves interrupted job state. It does
not launch an independent generation daemon. A currently executing model chunk
may take a short time to return before shutdown completes.

On Windows, the server is placed in a process job with kill-on-close enabled.
Force-ending the server also ends its FFmpeg children. When using Uvicorn
`--reload`, the worker watches the reloader's process sentinel and terminates its
process tree if that main process exits. Thus stopping the PyCharm run also stops
generation, including a forced stop of the reloader. This uses the Windows
[job object lifetime mechanism](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects).

A forced kill cannot save a final checkpoint; the next start recovers the last
flushed asset journal. Restart in PyCharm and use Studio's continue button to
resume. Completed audio is retained. Closing only the browser tab leaves the
server running; stop the PyCharm run or the Uvicorn console to stop generation.


## Startup resource planning and dynamic concurrency

Default concurrency is now **0 (automatic)**. The service samples per-core CPU
load for one second, respects CPU affinity and Linux cgroup v2 CPU/memory limits,
reads available RAM, and obtains free VRAM for the selected CUDA device. GPU
utilization is queried by UUID to respect CUDA device remapping. Unknown GPU
utilization uses at most one automatic slot.

The planner deducts existing workloads and reserves CPU capacity of 25%, RAM
of max(2GiB, 25% of total), and VRAM of max(1.5GiB, 25% of total). The settings are
`KOKORO_SERVE_RESOURCE_RESERVE_FRACTION`, `KOKORO_SERVE_RESERVE_MEMORY_MB`, and
`KOKORO_SERVE_RESERVE_VRAM_MB`. Each slot is budgeted at 1.5GiB VRAM and 768MiB
host RAM for CUDA, or 1.5GiB host RAM for CPU. Automatic slots are also limited by
CPU budget, GPU load and the service maximum of 8. These are conservative memory
estimates, not measured peak bounds for every text. Insufficient memory after
reserves produces a clear startup error. Auto device falls back to CPU when GPU
load reaches 75% or remaining VRAM cannot fit one estimated slot.

Automatic runs attempt below-normal CPU process priority so foreground work has
priority under contention. The health response reports whether this succeeded.
`resourcePlan` in `/health` and the startup log include the sampled resources,
reserved memory, actual device, concurrency and CPU threads. Batch workers use
the configured upper bound rather than the configured zero. The device and Torch
intra-op thread count remain fixed until restart. Explicit positive
concurrency/thread values override automatic selection and can consume more
resources than the computed budget. Audio quality and model precision are unchanged.

Dynamic admission is enabled by default (`KOKORO_SERVE_DYNAMIC_CONCURRENCY=true`).
Every `KOKORO_SERVE_RESOURCE_CHECK_SECONDS` (default 5 seconds, plus a one-second
CPU sample), it checks CPU load, available RAM and free VRAM. Two consecutive
samples below the memory reserves or above 90% CPU load reduce active concurrency
by one, down to one. Running audio completes normally; queued generation waits.
Three comfortable samples (CPU below 75%, memory above 115% of the reserves) and
a busy queue allow a one-slot expansion probe, up to the configured upper bound
(8 in automatic mode), including beyond the startup concurrency. Expansion must
fit the current CPU budget, host reserves and an extra slot allowance (1.5GiB
VRAM and 768MiB RAM for CUDA; 1.5GiB RAM for CPU). New independent models are loaded
and warmed in a worker thread before admission increases; allocation/warm-up time
is excluded from the next throughput sample. Allocation failures keep the current
limit and impose a cooldown. No models are loaded just because the service is idle.
The next two sufficiently populated samples must improve average completed text
characters per second by at least 5% against a baseline averaged over up to three
samples; otherwise the probe is reverted with a 60-sample
cooldown (about six minutes by default), avoiding repeated expensive model
reloads at a level that already failed to improve throughput.
This is a workload-dependent heuristic, not a guarantee of peak throughput.
GPU utilization is reported but does not alone reduce concurrency: the service's
own inference can legitimately saturate the GPU. Shrink retires idle surplus models;
busy surplus models retire after finishing their current audio. The primary model
stays resident, and released CUDA cache is returned at the next resource sample.
At minimum one slot continues; the controller
cannot guarantee foreground responsiveness under arbitrary external loads.
Use `false` for fixed concurrency. `/health` exposes current concurrency,
`concurrencyCapacity` (loaded models), `concurrencyCeiling` and `dynamicConcurrency`
samples/reasons. Batch status
reports current concurrency. Sampling failures keep the last limit, and the
monitor exits with the main service; no separate background service is spawned.

Compare warmed levels without starting a service or consuming real batch tasks:
`python -m scripts.benchmark --device auto --count 80 --sweep-max 4`.
Expansion stops when the reserved resource budget cannot fit another slot.


## Reusing local and Docker resources

Shared data stays in the project:

| Resource | Reuse |
| --- | --- |
| `data/huggingface` | Model weights, config and voice packs; bind-mounted into Docker |
| `data/wheels/common` | Portable Python wheels and the exported spaCy English model; read during Docker builds and local installs |
| `output` | Generated audio, batch tasks, journals and result packages; bind-mounted into Docker |
| Docker BuildKit caches | Persistent Linux pip and apt downloads across image rebuilds, including separate CPU/CUDA wheel URLs |
| Local pip cache | Existing Windows downloads are preserved; portable cached archives are copied into the common wheelhouse |

Activate the local Conda environment before using these helpers. To build while
preparing reusable resources automatically:

```powershell
.\scripts\docker-build.ps1              # CPU image
.\scripts\docker-build.ps1 -Gpu         # CUDA image
```

The helper exports installed spaCy model data, recoverable pure Python dependencies
and available portable cached wheels without downloading anything, builds the image,
then exports portable dependencies and language models installed by Docker back
into the shared wheelhouse using an offline,
short-lived container. It does not start the service. Existing Conda installs
can prepare the shared wheelhouse alone with:

```powershell
python -m scripts.prepare_shared_cache
```

Local dependency installs can reuse the same packages:

```powershell
.\scripts\install-local.ps1
```

Both scripts accept `-Python` to select an explicit interpreter. Native Windows
and Linux wheels (PyTorch, NumPy, etc.) remain platform-specific. Only wheels
marked pure Python and platform `any`, without native DLL/shared libraries, are
shared. The language model wheel preserves its spaCy compatibility range and
gets regenerated checksum records. Existing cached versions are reused when
compatible; a changed dependency/version may still require a download.

The Dockerfile keeps the large PyTorch install layer separate from application
requirements and source changes. pip/apt download caches are BuildKit mounts and
are not baked into the runtime image. Keep Docker builder caches and `data` to
retain downloads; deleting them or changing the platform/version may require
fetching resources again. See [Docker cache mounts](https://docs.docker.com/build/cache/optimize/).

Word audio is synthesized from explicit IPA through a strict English Misaki mapping, never from guessed spelling. Unsupported symbols, ambiguous optional forms and absent relationships are reported in the Studio review list and local generation report; successful targets can continue. This is target identity validation, not a guarantee of perceptual quality. Inspect representative generated recordings before publishing. Manifests use `typingo-audio-manifest/v2`; object keys include a pronunciation snapshot so different readings of one word cannot overwrite each other. Resume validates target identity, locale and checksums against the task.

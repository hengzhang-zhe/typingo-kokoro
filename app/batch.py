from __future__ import annotations

import asyncio
import hashlib
import gzip
import io
import json
import re
import shutil
import uuid
import zipfile
import time
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.voices import resolve_voice
from app.pronunciation import ipa_to_misaki

SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
OUTPUT_ROOT = Path("output") / "batches"


@dataclass
class BatchJob:
    id: str
    source_path: Path
    root: Path
    status: str = "imported"
    total: int = 0
    completed: int = 0
    failed: int = 0
    item_count: int = 0
    voices: dict[str, int] = field(default_factory=dict)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    zip_path: Path | None = None
    output_mode: str = "bundle"
    zip_paths: list[Path] = field(default_factory=list)
    concurrency: int = 1
    started_at: float | None = None
    initial_completed: int = 0
    elapsed_seconds: float = 0.0
    stop_event: threading.Event = field(default_factory=threading.Event)


_jobs: dict[str, BatchJob] = {}
_lock = asyncio.Lock()
_generation_lock = asyncio.Lock()
_generation_tasks: dict[asyncio.Task, BatchJob] = {}
_closing = False


def open_generation() -> None:
    global _closing
    _closing = False


def _runtime_concurrency() -> int:
    from app.engine import get_engine
    return getattr(get_engine(), 'concurrency', None) or get_settings().max_concurrency or 1


async def shutdown_generation() -> None:
    global _closing
    _closing = True
    owned = list(_generation_tasks.items())
    tasks = [task for task, _ in owned]
    for task, job in owned:
        job.stop_event.set()
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    for _, job in owned:
        # A task cancelled before its first turn never enters _run's finally.
        if job.status == "running":
            job.status = "interrupted"
            _save_status(job)


async def _thread_operation(function, *args):
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        except Exception:
            pass
        raise


def _check_job_running(job: BatchJob) -> None:
    if job.stop_event.is_set():
        raise RuntimeError("Batch interrupted by service shutdown")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe(value: str, field_name: str) -> str:
    if not isinstance(value,str) or not value or value in {".", ".."} or not SAFE_ID.fullmatch(value):
        raise ValueError(f"Invalid {field_name}: {value!r}")
    return value


def _read_task(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if data.get("schemaVersion") not in {"typingo-tts-export/v3"}:
        raise ValueError("Unsupported task schemaVersion")
    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError("items must be an array")
    return data


MAX_TASK_BYTES = 32 * 1024 * 1024
MAX_TASK_JSON_BYTES = 32 * 1024 * 1024
MAX_RESULT_PART_BYTES = 256 * 1024 * 1024
MAX_MANIFEST_PART_ASSETS = 5000


def decode_task(raw: bytes) -> dict[str, Any]:
    if len(raw) > MAX_TASK_BYTES:
        raise ValueError("Task file is too large")
    try:
        if raw.startswith(b"PK"):
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                files = [entry for entry in archive.infolist() if not entry.is_dir()]
                if len(files) != 1 or not files[0].filename.endswith(".json"):
                    raise ValueError("Task ZIP must contain exactly one JSON document")
                name = files[0].filename
                if name.startswith("/") or "\\" in name or ":" in name or ".." in name.split("/"):
                    raise ValueError("Unsafe ZIP path")
                if files[0].file_size > MAX_TASK_JSON_BYTES:
                    raise ValueError("Expanded task is too large")
                with archive.open(files[0]) as source:
                    raw = source.read(MAX_TASK_JSON_BYTES + 1)
        elif raw.startswith(b"\x1f\x8b"):
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as source:
                raw = source.read(MAX_TASK_JSON_BYTES + 1)
        if len(raw) > MAX_TASK_JSON_BYTES:
            raise ValueError("Expanded task is too large")
        data = json.loads(raw.decode("utf-8-sig"))
    except (OSError, zipfile.BadZipFile, UnicodeError) as exc:
        raise ValueError("Invalid task archive") from exc
    if not isinstance(data, dict) or data.get("schemaVersion") not in {"typingo-tts-export/v3"}:
        raise ValueError("Unsupported task schemaVersion")
    return data


async def import_task(raw: bytes) -> dict[str, Any]:
    data = decode_task(raw)
    items = data.get("items")
    if not isinstance(items, list) or len(items) > 100000:
        raise ValueError("items must be an array with at most 100000 reading targets")
    output_mode = data.get("outputMode", "bundle")
    if output_mode not in {"manifest", "bundle"}:
        raise ValueError("Unsupported outputMode")
    profiles = {}
    if not isinstance(data.get("voices", []), list):
        raise ValueError("voices must be an array")
    for profile in data.get("voices", []):
        if not isinstance(profile, dict):
            raise ValueError("voice profile must be an object")
        voice = _safe(str(profile.get("voice", "")), "voice")
        route = resolve_voice(voice)
        if profile.get("locale") != route.locale:
            raise ValueError(f"Invalid locale for voice={voice}")
        if voice in profiles:
            raise ValueError(f"Duplicate voice={voice}")
        profiles[voice] = profile
    seen_ids = set()
    skipped = data.get("skipped", [])
    if not isinstance(skipped, list) or any(not isinstance(x, dict) for x in skipped):
        raise ValueError("skipped must be an array of reports")
    skipped = list(skipped)

    voice_counter: Counter[str] = Counter()
    total = 0
    normalized_items = []

    for item in items:
        if not isinstance(item, dict):
            raise ValueError("item must be an object")
        content_id = _safe(str(item.get("contentId", "")), "contentId")
        target_identity = (content_id, item.get("pronunciationId"))
        if target_identity in seen_ids:
            raise ValueError(f"Duplicate contentId={content_id}")
        seen_ids.add(target_identity)
        text = str(item.get("text", "")).strip()
        if not text:
            raise ValueError(f"Blank text for contentId={content_id}")
        ids = item.get("voices")
        if not isinstance(ids, list) or not ids or any(not isinstance(v, str) for v in ids) or len(set(ids)) != len(ids):
            raise ValueError("voices must be a nonempty unique array")
        if any(v not in profiles for v in ids):
            raise ValueError("Unknown voice reference")
        variants = [profiles[v] for v in ids]
        pid = item.get("pronunciationId")
        snapshot = item.get("snapshot")
        if not isinstance(snapshot, str) or not re.fullmatch(r"[0-9a-f]{64}", snapshot):
            raise ValueError("Missing content snapshot")
        if pid is not None:
            _safe(pid, "pronunciationId")
            # Snapshot and voice validation still fail the whole malformed task.
            # Unsupported pronunciation notation is a reported target-level omission.
            if any(v["locale"] != item["locale"] for v in variants):
                raise ValueError("Voice locale differs from target pronunciation")
        elif item.get("ipa") is not None or item.get("locale") is not None:
            raise ValueError("IPA requires a pronunciation target")
        expected = hashlib.sha256("\0".join([text, pid or "", item.get("locale") or "", item.get("ipa") or ""]).encode()).hexdigest()
        if expected != snapshot:
            raise ValueError("Content snapshot mismatch")

        if pid is not None:
            try:
                ipa_to_misaki(item.get("ipa"), item.get("locale"))
            except ValueError as exc:
                skipped.append({"contentId": content_id, "pronunciationId": pid, "text": text, "reason": str(exc)})
                continue

        normalized_variants = []
        for variant in variants:
            if not isinstance(variant, dict):
                raise ValueError("variant must be an object")
            voice = _safe(str(variant.get("voice", "")), "voice")
            route = resolve_voice(voice)
            locale = str(variant.get("locale") or route.locale)
            if locale != route.locale:
                raise ValueError(f"Invalid locale for voice={voice}")
            normalized_variants.append({
                "voice": voice,
                "locale": locale,
            })
            profiles[voice] = {"voice": voice, "locale": locale}
            voice_counter[voice] += 1
            total += 1

        if normalized_variants:
            normalized_items.append({
                "contentId": content_id,
                "text": text,
                "voices": [v["voice"] for v in normalized_variants],
                "pronunciationId": pid, "locale": item.get("locale"), "ipa": item.get("ipa"), "snapshot": snapshot,
            })

    if len({cid for cid, _ in seen_ids}) > 10000:
        raise ValueError("At most 10000 distinct content items per task")

    job_id = uuid.uuid4().hex
    root = OUTPUT_ROOT / job_id
    root.mkdir(parents=True, exist_ok=False)
    source_path = root / "task.json"
    normalized = {"schemaVersion": "typingo-tts-export/v3", "outputMode": output_mode, "voices": list(profiles.values()), "items": normalized_items, "skipped": skipped}
    source_path.write_text(json.dumps(normalized, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    job = BatchJob(
        id=job_id,
        source_path=source_path,
        root=root,
        total=total,
        item_count=len(normalized_items),
        skipped=skipped,
        voices=dict(voice_counter),
        output_mode=output_mode,
    )
    async with _lock:
        _jobs[job_id] = job
    _save_status(job)

    return job_view(job)


def job_view(job: BatchJob) -> dict[str, Any]:
    elapsed = time.monotonic() - job.started_at if job.started_at is not None else job.elapsed_seconds
    processed = job.completed + job.failed - job.initial_completed
    rate = processed / elapsed if elapsed > 0 else 0
    return {
        "concurrency": _runtime_concurrency() if job.status == 'running' else job.concurrency,
        "elapsedSeconds": round(elapsed, 1),
        "initialCompleted": job.initial_completed,
        "audioPerSecond": round(rate, 3),
        "etaSeconds": round(max(0, job.total - job.completed - job.failed) / rate) if rate > 0 else None,
        "id": job.id,
        "status": job.status,
        "outputMode": job.output_mode,
        "downloadParts": len(job.zip_paths) if job.zip_paths else (1 if job.zip_path else 0),
        "itemCount": job.item_count,
        "skippedCount": len(job.skipped),
        "skipped": job.skipped,
        "audioCount": job.total,
        "completed": job.completed,
        "failed": job.failed,
        "voices": job.voices,
        "error": job.error,
        "downloadReady": job.zip_path is not None and job.zip_path.exists(),
    }


def get_job(job_id: str) -> BatchJob:
    _safe(job_id, "jobId")
    job = _jobs.get(job_id)
    if job is None:
        root = OUTPUT_ROOT / job_id
        source = root / "task.json"
        status_file = root / "status.json"
        if source.exists() and status_file.exists():
            saved = json.loads(status_file.read_text(encoding="utf-8"))
            job = BatchJob(
                id=job_id,
                source_path=source,
                root=root,
                status="interrupted" if saved.get("status") == "running" else saved.get("status", "unknown"),
                concurrency=int(saved.get("concurrency", 1)),
                initial_completed=int(saved.get("initialCompleted", 0)),
                elapsed_seconds=float(saved.get("elapsedSeconds", 0)),
                total=int(saved.get("audioCount", 0)),
                completed=int(saved.get("completed", 0)),
                failed=int(saved.get("failed", 0)),
                item_count=int(saved.get("itemCount", 0)),
                skipped=saved.get("skipped", []),
                voices=saved.get("voices", {}),
                error=saved.get("error"),
                output_mode=saved.get("outputMode", "bundle"),
                zip_paths=sorted(root.glob("typingo-audio-part-*.zip")),
                zip_path=(root / "typingo-audio-batch.zip") if (root / "typingo-audio-batch.zip").exists() else None,
            )
            if job.zip_paths:
                job.zip_path = job.zip_paths[0]
            _jobs[job_id] = job
    if job is None:
        raise KeyError(job_id)
    return job


def _save_status(job: BatchJob) -> None:
    temporary = job.root / "status.tmp"
    temporary.write_text(
        json.dumps(job_view(job), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(job.root / "status.json")


async def start_generation(job_id: str) -> dict[str, Any]:
    if _closing:
        raise ValueError("Kokoro service is shutting down")
    job = get_job(job_id)
    if job.status == "running":
        return job_view(job)
    if job.status not in {"imported", "failed", "completed_with_errors", "interrupted"}:
        raise ValueError(f"Job cannot be started from status={job.status}")

    job.status = "running"
    job.stop_event.clear()
    job.concurrency = _runtime_concurrency()
    job.initial_completed = job.completed
    job.elapsed_seconds = 0
    job.failed = 0
    job.error = None
    job.zip_path = None
    job.zip_paths = []

    # Successful audio and the append-only asset journal survive retries.
    for stale in ("audio-manifest.json", "generation-report.json", "typingo-audio-batch.zip", "typingo-audio-only.zip"):
        p = job.root / stale
        if p.exists():
            p.unlink()

    for stale_audio in job.root.glob("typingo-audio-only-*.zip"):
        stale_audio.unlink()
    for part in job.root.glob("typingo-audio-part-*.zip"):
        part.unlink()
    _save_status(job)
    task = asyncio.create_task(_run(job), name=f"batch-{job.id}")
    _generation_tasks[task] = job
    task.add_done_callback(lambda completed: _generation_tasks.pop(completed, None))
    return job_view(job)


async def _run(job: BatchJob) -> None:
    try:
        async with _generation_lock:
            await _run_one(job)
    except asyncio.CancelledError:
        job.status = "interrupted"
        _save_status(job)
        raise
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        _save_status(job)



def _recover_assets(job: BatchJob, assets_file: Path) -> set[tuple[str, str | None, str]]:
    """Discard incomplete journal tails and retain only intact audio files."""
    task = _read_task(job.source_path)
    expected_targets = {(i["contentId"], i.get("pronunciationId"), v): i for i in task["items"] for v in i["voices"]}
    profiles = {v["voice"]: v for v in task["voices"]}
    recovered = set()
    temporary = assets_file.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        if assets_file.exists():
            with assets_file.open(encoding="utf-8") as source:
                for line in source:
                    _check_job_running(job)
                    try:
                        asset = json.loads(line)
                        content_id = _safe(asset["contentId"], "contentId")
                        voice = _safe(asset["voice"], "voice")
                        pid = asset.get("pronunciationId")
                        key = (content_id, pid, voice)
                        item = expected_targets.get(key)
                        if item is None or any(asset.get(k) != item.get(k) for k in ("snapshot", "ipa", "pronunciationId")):
                            continue
                        if asset.get("locale") != profiles[voice]["locale"]:
                            continue
                        if asset.get("role") != ("pronunciation" if pid is not None else "primary"):
                            continue
                        if not re.fullmatch(r"[0-9a-f]{64}", asset["snapshot"]):
                            continue
                        expected = Path("audio") / "kokoro" / content_id / asset["snapshot"] / f"{voice}.mp3"
                        target = job.root / expected
                        if asset["objectKey"] != expected.as_posix() or key in recovered:
                            continue
                        if not target.is_file() or target.stat().st_size != asset["sizeBytes"]:
                            continue
                        if _file_digest(target) != asset["checksumSha256"]:
                            continue
                        output.write(json.dumps(asset, separators=(",", ":")) + "\n")
                        recovered.add(key)
                    except (ValueError, KeyError, TypeError, OSError):
                        continue
    temporary.replace(assets_file)
    return recovered


def _file_digest(target: Path) -> str:
    digest = hashlib.sha256()
    with target.open("rb") as source:
        while chunk := source.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


async def _run_one(job: BatchJob) -> None:
    assets_file = job.root / "assets.jsonl"
    failures_file = job.root / "failures.jsonl"
    recovered = await _thread_operation(_recover_assets, job, assets_file)
    job.completed = len(recovered)
    job.initial_completed = job.completed
    job.failed = 0
    job.concurrency = _runtime_concurrency()
    job.started_at = time.monotonic()
    assets_stream = assets_file.open("a", encoding="utf-8")
    failures_stream = failures_file.open("w", encoding="utf-8")
    try:
        task = _read_task(job.source_path)
        from app.engine import get_engine
        engine = get_engine()

        profiles = {v["voice"]: v for v in task.get("voices", [])}
        def pending():
            for item in task["items"]:
                for variant in [profiles[v] for v in item["voices"]]:
                    if (item["contentId"], item.get("pronunciationId"), variant["voice"]) not in recovered:
                        yield item, variant

        tasks = iter(pending())
        last_checkpoint = time.monotonic()
        async def worker():
            nonlocal last_checkpoint
            while True:
                try:
                    item, variant = next(tasks)
                    content_id, text = item["contentId"], item["text"]
                except StopIteration:
                    return
                voice = variant["voice"]
                locale = variant["locale"]
                try:
                    relative = Path("audio") / "kokoro" / content_id / item["snapshot"] / f"{voice}.mp3"
                    target = job.root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if hasattr(engine, "synthesize_to_file"):
                        options = {"phonemes": ipa_to_misaki(item["ipa"], locale)} if item.get("pronunciationId") else {}
                        await engine.synthesize_to_file(text=text, voice=voice, speed=1.0, target=target, **options)
                    else:
                        if item.get("pronunciationId"):
                            raise ValueError("Engine does not support explicit pronunciation synthesis")
                        data = await engine.synthesize(text=text, voice=voice, speed=1.0, response_format="mp3")
                        target.write_bytes(data)
                    size_bytes = target.stat().st_size
                    if size_bytes > 100 * 1024 * 1024:
                        target.unlink()
                        raise ValueError("Single audio exceeds Typingo's 100MB object limit")
                    digest = await asyncio.to_thread(_file_digest, target)
                    assets_stream.write(json.dumps({
                        "contentId": content_id,
                        "role": "pronunciation" if item.get("pronunciationId") else "primary",
                        "pronunciationId": item.get("pronunciationId"), "ipa": item.get("ipa"), "snapshot": item["snapshot"],
                        "locale": locale,
                        "voice": voice,
                        "objectKey": relative.as_posix(),
                        "checksumSha256": digest,
                        "sizeBytes": size_bytes,
                    }, separators=(",", ":")) + "\n")
                    assets_stream.flush()
                    job.completed += 1
                except Exception as exc:
                    target.unlink(missing_ok=True)
                    job.failed += 1
                    failures_stream.write(json.dumps({
                        "contentId": content_id,
                        "voice": voice,
                        "message": str(exc)[:4096],
                    }, ensure_ascii=False) + "\n")
                finally:
                    if time.monotonic() - last_checkpoint >= 1:
                        _save_status(job)
                        last_checkpoint = time.monotonic()


        # Only N worker tasks exist, even for hundreds of thousands of outputs.
        async with asyncio.TaskGroup() as group:
            workers = (getattr(engine, 'concurrency_ceiling', job.concurrency)
                       if getattr(getattr(engine, 'settings', None), 'dynamic_concurrency', False)
                       else getattr(engine, 'capacity', job.concurrency))
            for _ in range(workers):
                group.create_task(worker())

        assets_stream.close()
        failures_stream.close()
        await _thread_operation(_package_results, job, assets_file)
        with (job.root / "generation-report.json").open("w", encoding="utf-8") as report:
            report.write(json.dumps({"generatedAt": _now(), "requested": job.total, "completed": job.completed, "failed": job.failed, "skipped": job.skipped}, ensure_ascii=False)[:-1] + ',"failures":[')
            with failures_file.open(encoding="utf-8") as failures:
                first = True
                for line in failures:
                    if not first: report.write(",")
                    report.write(line.strip()); first = False
            report.write("]}")
        job.status = "completed_with_errors" if job.failed or job.skipped else "completed"
    except asyncio.CancelledError:
        job.status = "interrupted"
        raise
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
    finally:
        assets_stream.close()
        failures_stream.close()
        job.elapsed_seconds = time.monotonic() - job.started_at
        job.started_at = None
        _save_status(job)


def _package_results(job: BatchJob, assets_file: Path) -> None:
    """Every part is a complete exchange ZIP, never a split ZIP byte fragment."""
    parts = []
    selected = []
    size = 0
    def flush():
        nonlocal selected, size
        path = job.root / f"typingo-audio-part-{len(parts) + 1:04d}.zip"
        manifest = {"schemaVersion": "typingo-audio-manifest/v2", "generatedAt": _now(),
                    "provider": "kokoro", "model": "hexgrad/Kokoro-82M", "assets": selected}
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            if job.output_mode == "bundle":
                for asset in selected:
                    _check_job_running(job)
                    archive.write(job.root / asset["objectKey"], asset["objectKey"], compress_type=zipfile.ZIP_STORED)
            archive.writestr("audio-manifest.json", json.dumps(manifest, ensure_ascii=False, separators=(",", ":")))
        parts.append(path)
        selected = []; size = 0
    with assets_file.open(encoding="utf-8") as source:
        for line in source:
            _check_job_running(job)
            asset = json.loads(line)
            added = asset["sizeBytes"] if job.output_mode == "bundle" else 0
            if selected and (size + added > MAX_RESULT_PART_BYTES or len(selected) >= MAX_MANIFEST_PART_ASSETS):
                flush()
            selected.append(asset); size += added
    if selected or not parts: flush()
    job.zip_paths = parts
    job.zip_path = parts[0]
    if len(parts) == 1:
        target = job.root / "typingo-audio-batch.zip"
        shutil.copyfile(parts[0], target)
        job.zip_path = target
    # Full local manifest remains useful for manual audio processing, streamed from disk.
    with (job.root / "audio-manifest.json").open("w", encoding="utf-8") as output:
        output.write(json.dumps({"schemaVersion": "typingo-audio-manifest/v2", "generatedAt": _now(), "provider": "kokoro", "model": "hexgrad/Kokoro-82M"})[:-1] + ',"assets":[')
        with assets_file.open(encoding="utf-8") as source:
            first = True
            for line in source:
                if not first: output.write(",")
                output.write(line.strip()); first = False
        output.write("]}")

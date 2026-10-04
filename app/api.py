import json
import io
import zipfile
import uuid
import asyncio
import shutil
from pathlib import Path
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from app.audio import MEDIA_TYPES
from app.batch import MAX_TASK_BYTES, get_job, import_task, job_view, start_generation
from app.config import get_settings
from app.engine import get_engine
from app.schemas import SpeechRequest
from app.voices import resolve_voice, voice_catalog

router = APIRouter()
settings = get_settings()

@router.get("/health")
def health():
    engine = get_engine()
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
        "model_repo": settings.model_repo,
        "device": engine.device,
        "requestedDevice": settings.device,
        "concurrency": engine.concurrency,
        "concurrencyCapacity": engine.capacity,
        "concurrencyCeiling": engine.concurrency_ceiling,
        "dynamicConcurrency": engine.dynamic_status,
        "resourcePlan": engine.resource_plan.view() if engine.resource_plan else None,
        "belowNormalPriority": engine.below_normal_priority,
        "mp3Quality": settings.mp3_quality,
        "loaded": engine.model is not None,
    }

@router.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [{
            "id": "kokoro",
            "object": "model",
            "owned_by": "hexgrad",
            "source": settings.model_repo,
        }],
    }

@router.get("/v1/audio/voices")
def voices():
    data = voice_catalog()
    return {
        "schemaVersion": "kokoro-voice-catalog/v1",
        "source": settings.model_repo,
        "count": len(data),
        "voices": data,
    }

@router.get("/v1/audio/voices/download")
def download_voices():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("voice-catalog.json", json.dumps(voices(), ensure_ascii=False, separators=(",", ":")))
    return Response(buffer.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": "attachment; filename=voice-catalog.zip"})

@router.post("/v1/audio/speech")
async def speech(request: SpeechRequest):
    if request.model not in {"kokoro", "Kokoro-82M", settings.model_repo}:
        raise HTTPException(
            status_code=400,
            detail="Only the Kokoro model is supported",
        )

    text = request.input.strip()
    if not text:
        raise HTTPException(status_code=400, detail="input cannot be blank")

    if len(text) > settings.max_input_chars:
        raise HTTPException(
            status_code=413,
            detail=f"input exceeds {settings.max_input_chars} characters",
        )

    try:
        resolve_voice(request.voice)
        data = await get_engine().synthesize(
            text=text,
            voice=request.voice,
            speed=request.speed,
            response_format=request.response_format,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"TTS synthesis failed: {exc}",
        ) from exc

    return Response(
        content=data,
        media_type=MEDIA_TYPES[request.response_format],
    )


@router.post("/v1/batches/import")
async def import_batch(file: UploadFile = File(...)):
    try:
        raw = await file.read(MAX_TASK_BYTES + 1)
        return await import_task(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/v1/batches")
def recent_batches():
    from app.batch import OUTPUT_ROOT
    paths = sorted(OUTPUT_ROOT.glob("*/status.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:10]
    return {"jobs": [job_view(get_job(path.parent.name)) for path in paths]}


@router.post("/v1/batches/{job_id}/generate")
async def generate_batch(job_id: str):
    try:
        return await start_generation(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Batch job not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/v1/batches")
def recent_batches():
    from app.batch import OUTPUT_ROOT
    paths = sorted(OUTPUT_ROOT.glob("*/status.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:10]
    return {"jobs": [job_view(get_job(path.parent.name)) for path in paths]}


@router.get("/v1/batches")
def recent_batches():
    from app.batch import OUTPUT_ROOT
    paths = sorted(OUTPUT_ROOT.glob("*/status.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:10]
    return {"jobs": [job_view(get_job(path.parent.name)) for path in paths]}


@router.get("/v1/batches/{job_id}")
def batch_status(job_id: str):
    try:
        return job_view(get_job(job_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Batch job not found") from exc


@router.get("/v1/batches/{job_id}/download")
def batch_download(job_id: str, part: int = 1):
    try:
        job = get_job(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Batch job not found") from exc
    if job.zip_path is None or not job.zip_path.exists():
        raise HTTPException(status_code=409, detail="Batch output is not ready")
    paths = job.zip_paths or [job.zip_path]
    if part < 1 or part > len(paths):
        raise HTTPException(status_code=400, detail="Invalid result part")
    return FileResponse(
        paths[part - 1],
        media_type="application/zip",
        filename=f"typingo-audio-batch-{job.id}-part-{part:04d}.zip",
    )


@router.get("/v1/batches/{job_id}/audio/download")
def batch_audio_download(job_id: str, part: int = 1):
    try:
        job = get_job(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Batch job not found") from exc
    if job.status not in {"completed", "completed_with_errors"}:
        raise HTTPException(status_code=409, detail="Batch output is not ready")
    paths = job.zip_paths or [job.zip_path]
    if part < 1 or part > len(paths):
        raise HTTPException(status_code=400, detail="Invalid result part")
    target = job.root / f"typingo-audio-only-{part:04d}.zip"
    if not target.exists():
        temporary = job.root / f"typingo-audio-only-{uuid.uuid4().hex}.tmp"
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            with zipfile.ZipFile(paths[part - 1]) as result:
                manifest = json.loads(result.read("audio-manifest.json"))
            for asset in manifest["assets"]:
                archive.write(job.root / asset["objectKey"], asset["objectKey"], compress_type=zipfile.ZIP_STORED)
        temporary.replace(target)
    return FileResponse(target, media_type="application/zip", filename=f"typingo-audio-only-{job.id}-part-{part:04d}.zip")


_upload_lock = asyncio.Lock()

def upload_root(upload_id: str) -> Path:
    try:
        if str(uuid.UUID(upload_id)) != upload_id: raise ValueError()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid upload ID") from exc
    from app import batch
    return batch.OUTPUT_ROOT.parent / "uploads" / upload_id

@router.post("/v1/batches/import/chunks")
async def import_task_chunk(uploadId: str = Form(...), index: int = Form(...), totalChunks: int = Form(...),
                            totalSize: int = Form(...), file: UploadFile = File(...)):
    if totalSize < 1 or totalSize > MAX_TASK_BYTES or totalChunks < 1 or totalChunks > 4096 or not 0 <= index < totalChunks:
        raise HTTPException(status_code=400, detail="Invalid task chunk metadata; maximum task is 32MB")
    root = upload_root(uploadId)
    async with _upload_lock:
        root.mkdir(parents=True, exist_ok=True)
        metadata = root / "metadata.json"
        expected = {"totalSize": totalSize, "totalChunks": totalChunks}
        if metadata.exists() and json.loads(metadata.read_text()) != expected:
            raise HTTPException(status_code=400, detail="Upload metadata changed")
        metadata.write_text(json.dumps(expected))
        temporary = root / f"{index}.tmp"
        try:
            size = 0
            with temporary.open("wb") as output:
                while data := await file.read(65536):
                    size += len(data)
                    if size > 4 * 1024 * 1024:
                        raise HTTPException(status_code=413, detail="Chunk exceeds 4MB")
                    output.write(data)
            if size == 0: raise HTTPException(status_code=400, detail="Empty chunk")
            temporary.replace(root / f"{index}.part")
        finally:
            temporary.unlink(missing_ok=True)
    return {"uploaded": index}

@router.post("/v1/batches/import/chunks/complete")
async def complete_task_chunks(request: dict):
    root = upload_root(str(request.get("uploadId", "")))
    async with _upload_lock:
        metadata = root / "metadata.json"
        if not metadata.exists(): raise HTTPException(status_code=400, detail="Upload session not found")
        saved = json.loads(metadata.read_text())
        if saved != {"totalSize": request.get("totalSize"), "totalChunks": request.get("totalChunks")}:
            raise HTTPException(status_code=400, detail="Upload metadata changed")
        paths = [root / f"{index}.part" for index in range(saved["totalChunks"])]
        if not all(path.is_file() for path in paths): raise HTTPException(status_code=400, detail="Missing task chunk")
        try:
            assembled = root / "task.upload"
            size = 0
            with assembled.open("wb") as output:
                for path in paths:
                    with path.open("rb") as source:
                        while data := source.read(65536):
                            size += len(data)
                            if size > saved["totalSize"]: raise ValueError("Chunk size mismatch")
                            output.write(data)
            if size != saved["totalSize"]: raise ValueError("Chunk size mismatch")
            return await import_task(assembled.read_bytes())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            shutil.rmtree(root)

@router.delete("/v1/batches/import/chunks/{upload_id}")
async def cancel_task_chunks(upload_id: str):
    async with _upload_lock:
        root = upload_root(upload_id)
        if root.exists(): shutil.rmtree(root)
    return {"cancelled": True}

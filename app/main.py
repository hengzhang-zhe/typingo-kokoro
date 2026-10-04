from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api import router
from app.config import get_settings
from app.engine import get_engine
from app.batch import open_generation, shutdown_generation
from app.process_lifetime import bind_process_lifetime

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    bind_process_lifetime()
    engine = get_engine()
    engine.load()
    engine.start_resource_monitor()
    open_generation()
    try:
        yield
    finally:
        engine.request_shutdown()
        await engine.stop_resource_monitor()
        await shutdown_generation()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Self-hosted API service for official Kokoro inference.",
    lifespan=lifespan,
)

app.include_router(router)

@app.get("/", include_in_schema=False)
def batch_studio():
    return FileResponse(Path("web/index.html"))

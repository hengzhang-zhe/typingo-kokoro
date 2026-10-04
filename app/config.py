from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    app_name: str = "Typingo Kokoro"
    app_version: str = "0.1.0"

    model_repo: str = "hexgrad/Kokoro-82M"
    device: str = "auto"

    max_input_chars: int = 4096
    max_concurrency: int = Field(default=0, ge=0, le=8)
    torch_num_threads: int = Field(default=0, ge=0, le=256)
    dynamic_concurrency: bool = True
    resource_check_seconds: float = Field(default=5, ge=2, le=60)
    resource_reserve_fraction: float = Field(default=0.25, ge=0.1, le=0.75)
    reserve_memory_mb: int = Field(default=2048, ge=256)
    reserve_vram_mb: int = Field(default=1536, ge=256)
    mp3_quality: int = Field(default=0, ge=0, le=9)
    sample_rate: int = Field(default=24000, ge=24000, le=24000)

    model_config = SettingsConfigDict(
        env_prefix="KOKORO_SERVE_",
        env_file=".env",
        extra="ignore",
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()

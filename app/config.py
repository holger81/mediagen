"""Portainer-friendly settings (env overrides, no required .env file)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    cache_dir: Path = Path("/cache")
    cache_max_items: int = 1000
    comfyui_base_url: str = "http://192.168.10.31:8188"
    workflows_dir: Path = Path("/app/workflows")
    outpaint_poll_interval_s: float = 1.5
    outpaint_poll_timeout_s: float = 180.0
    outpaint_retry_after_s: int = 5
    cors_origins: str = "*"
    # Empty = open admin UI (LAN trust). Set to require ?token= or X-Admin-Token.
    admin_token: str = ""
    admin_log_capacity: int = 500


@lru_cache
def get_settings() -> Settings:
    return Settings()

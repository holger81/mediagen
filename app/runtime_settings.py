"""Persist small runtime admin settings under CACHE_DIR."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_SETTINGS_NAME = "mediagen_runtime.json"


def runtime_settings_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / _SETTINGS_NAME


def load_runtime_settings(cache_dir: Path) -> dict:
    path = runtime_settings_path(cache_dir)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError) as exc:
        logger.warning("Could not read runtime settings %s: %s", path, exc)
        return {}


def save_runtime_settings(cache_dir: Path, data: dict) -> None:
    path = runtime_settings_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def apply_runtime_overrides(cache_dir: Path, *, flux_quality_gate: bool) -> bool:
    """Merge env default with persisted override. Returns effective gate value."""
    stored = load_runtime_settings(cache_dir)
    if "flux_quality_gate" in stored:
        return bool(stored["flux_quality_gate"])
    return flux_quality_gate

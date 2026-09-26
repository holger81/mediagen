"""In-memory activity for live admin: current requests + Comfy payloads."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class JobActivity:
    hash: str
    phase: str  # queued|uploading|prompt|waiting|downloading|gating|done|error
    src_w: int = 0
    src_h: int = 0
    pads: str = ""
    out_size: str = ""
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    # What we sent / got from Comfy
    comfy_image: str | None = None
    prompt_id: str | None = None
    seed: int | None = None
    positive_prompt: str | None = None
    negative_prompt: str | None = None
    client_id: str | None = None
    output_filename: str | None = None
    error: str | None = None
    quality_gate: bool | None = None

    def touch(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.updated_at = time.time()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ActivityTracker:
    """Thread-safe current + recent job activity for the admin UI."""

    def __init__(self, *, recent_limit: int = 30) -> None:
        self._lock = threading.RLock()
        self._current: dict[str, JobActivity] = {}
        self._recent: deque[JobActivity] = deque(maxlen=max(5, recent_limit))

    def start(
        self,
        content_hash: str,
        *,
        src_w: int,
        src_h: int,
        pads: str,
        out_size: str,
        quality_gate: bool,
    ) -> JobActivity:
        job = JobActivity(
            hash=content_hash,
            phase="queued",
            src_w=src_w,
            src_h=src_h,
            pads=pads,
            out_size=out_size,
            quality_gate=quality_gate,
        )
        with self._lock:
            self._current[content_hash] = job
        return job

    def update(self, content_hash: str, **kwargs: Any) -> None:
        with self._lock:
            job = self._current.get(content_hash)
            if job is None:
                return
            job.touch(**kwargs)

    def finish(self, content_hash: str, *, phase: str = "done", error: str | None = None) -> None:
        with self._lock:
            job = self._current.pop(content_hash, None)
            if job is None:
                return
            job.touch(phase=phase, error=error)
            self._recent.appendleft(job)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            current = [j.as_dict() for j in self._current.values()]
            recent = [j.as_dict() for j in self._recent]
        current.sort(key=lambda j: j.get("started_at") or 0, reverse=True)
        return {"current": current, "recent": recent}

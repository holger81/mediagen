"""Outpaint orchestration: cache hit sync; miss starts background job (202/poll)."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from app.cache import MediaCache
from app.comfy_client import ComfyUiOutpaintClient
from app.local_pad import accept_flux_pad, has_uniform_edges, pad_from_edges

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OutpaintResult:
    hash: str
    bytes: bytes
    source: str  # flux | local
    cache: str  # hit | miss


@dataclass(frozen=True)
class GeneratingStatus:
    hash: str
    retry_after_s: int
    status: str = "generating"


class OutpaintService:
    def __init__(
        self,
        cache: MediaCache,
        comfy: ComfyUiOutpaintClient,
        *,
        retry_after_s: int = 5,
    ) -> None:
        self.cache = cache
        self.comfy = comfy
        self.retry_after_s = max(1, retry_after_s)
        self._inflight: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

    def _hit_result(self, content_hash: str, *, touch: bool) -> OutpaintResult | None:
        hit = self.cache.get_by_hash(content_hash, touch=touch)
        if hit is None:
            return None
        data = hit.path.read_bytes()
        if not data:
            return None
        return OutpaintResult(
            hash=hit.hash,
            bytes=data,
            source=hit.source,
            cache="hit",
        )

    def is_generating(self, content_hash: str) -> bool:
        task = self._inflight.get(content_hash)
        return task is not None and not task.done()

    def _settled_cache_hit(
        self,
        content_hash: str,
        source_bytes: bytes,
        *,
        touch: bool,
    ) -> OutpaintResult | None:
        """Return a finished cache entry — never re-queue the same cover forever.

        Flux results are always final. Local pads are final for uniform mattes,
        and also final after a generation attempt (`.done` marker) so a rejected
        or timed-out Flux does not restart on every playlist play.
        """
        hit = self._hit_result(content_hash, touch=touch)
        if hit is None:
            return None
        if hit.source == "flux":
            return hit
        if has_uniform_edges(source_bytes):
            return hit
        if self.cache.is_done(content_hash):
            return hit
        # Legacy pictorial local without `.done`: keep it (do not wipe). One
        # Comfy attempt already happened when it was written.
        logger.info(
            "Keeping settled local pad for pictorial cover %s (no re-queue)",
            content_hash[:12],
        )
        self.cache.mark_done(content_hash)
        return hit

    async def submit(self, source_bytes: bytes) -> OutpaintResult | GeneratingStatus:
        """Cache hit → result. Uniform local-only → sync result. Else start job → generating."""
        if not source_bytes:
            raise ValueError("empty image")
        content_hash = self.cache.hash_for(source_bytes)

        hit = self._settled_cache_hit(content_hash, source_bytes, touch=True)
        if hit is not None:
            return hit

        async with self._lock:
            hit = self._settled_cache_hit(content_hash, source_bytes, touch=True)
            if hit is not None:
                return hit
            if self.is_generating(content_hash):
                return GeneratingStatus(
                    hash=content_hash,
                    retry_after_s=self.retry_after_s,
                )

            # Fast path: uniform/black mattes only need local pad — finish sync.
            if has_uniform_edges(source_bytes):
                result = await self._generate(source_bytes, content_hash)
                return result

            task = asyncio.create_task(
                self._run_job(content_hash, source_bytes),
                name=f"outpaint-{content_hash[:12]}",
            )
            self._inflight[content_hash] = task
            return GeneratingStatus(
                hash=content_hash,
                retry_after_s=self.retry_after_s,
            )

    def lookup(self, content_hash: str) -> OutpaintResult | GeneratingStatus | None:
        """Ready result, generating status, or None if unknown."""
        hit = self._hit_result(content_hash, touch=True)
        if hit is not None:
            return hit
        if self.is_generating(content_hash):
            return GeneratingStatus(
                hash=content_hash,
                retry_after_s=self.retry_after_s,
            )
        return None

    async def _run_job(self, content_hash: str, source_bytes: bytes) -> None:
        try:
            await self._generate(source_bytes, content_hash)
        except Exception:
            logger.exception("Outpaint job failed for %s", content_hash[:12])
        finally:
            async with self._lock:
                current = self._inflight.get(content_hash)
                if current is asyncio.current_task():
                    self._inflight.pop(content_hash, None)

    async def _generate(self, source_bytes: bytes, content_hash: str) -> OutpaintResult:
        hit = self._hit_result(content_hash, touch=False)
        if hit is not None and (
            hit.source == "flux"
            or has_uniform_edges(source_bytes)
            or self.cache.is_done(content_hash)
        ):
            return hit

        local = await asyncio.to_thread(pad_from_edges, source_bytes)
        if local is None:
            raise RuntimeError("local pad failed")

        source = "local"
        result_bytes = local

        if not has_uniform_edges(source_bytes):
            flux = await self.comfy.outpaint(source_bytes)
            accepted = accept_flux_pad(flux, source_bytes) if flux else None
            if accepted:
                result_bytes = accepted
                source = "flux"
            elif flux:
                logger.info("Rejected Flux pad for %s; keeping local", content_hash[:12])

        stored = self.cache.put_by_hash(content_hash, result_bytes, source=source)
        if stored is None:
            raise RuntimeError("cache write failed")
        if source == "flux":
            (self.cache.cache_dir / f"{content_hash}.flux").touch()
        # Always mark done so pictorial local fallbacks are not re-queued forever.
        self.cache.mark_done(content_hash)

        return OutpaintResult(
            hash=content_hash,
            bytes=result_bytes,
            source=source,
            cache="miss",
        )

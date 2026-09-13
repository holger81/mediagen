"""Outpaint orchestration: cache → local pad → optional Comfy Flux upgrade."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from app.cache import MediaCache
from app.comfy_client import ComfyUiOutpaintClient
from app.local_pad import has_uniform_edges, pad_from_edges, should_reject_flux_pad

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OutpaintResult:
    hash: str
    bytes: bytes
    source: str  # flux | local
    cache: str  # hit | miss


class OutpaintService:
    def __init__(self, cache: MediaCache, comfy: ComfyUiOutpaintClient) -> None:
        self.cache = cache
        self.comfy = comfy
        self._inflight: dict[str, asyncio.Future[OutpaintResult]] = {}
        self._lock = asyncio.Lock()

    async def outpaint(self, source_bytes: bytes) -> OutpaintResult:
        if not source_bytes:
            raise ValueError("empty image")
        content_hash = self.cache.hash_for(source_bytes)

        hit = self.cache.get(source_bytes, touch=True)
        if hit is not None:
            data = hit.path.read_bytes()
            if data:
                return OutpaintResult(
                    hash=hit.hash, bytes=data, source=hit.source, cache="hit"
                )

        async with self._lock:
            existing = self._inflight.get(content_hash)
            if existing is not None:
                waiter: asyncio.Future[OutpaintResult] = asyncio.get_running_loop().create_future()

                def _forward(fut: asyncio.Future[OutpaintResult]) -> None:
                    if fut.cancelled():
                        waiter.cancel()
                        return
                    exc = fut.exception()
                    if exc is not None:
                        waiter.set_exception(exc)
                    else:
                        waiter.set_result(fut.result())

                existing.add_done_callback(_forward)
                return await waiter

            loop = asyncio.get_running_loop()
            future: asyncio.Future[OutpaintResult] = loop.create_future()
            self._inflight[content_hash] = future

        try:
            result = await self._generate(source_bytes, content_hash)
            if not future.done():
                future.set_result(result)
            return result
        except Exception as exc:
            if not future.done():
                future.set_exception(exc)
            raise
        finally:
            async with self._lock:
                self._inflight.pop(content_hash, None)

    async def _generate(self, source_bytes: bytes, content_hash: str) -> OutpaintResult:
        # Re-check after waiting for the lock / race.
        hit = self.cache.get_by_hash(content_hash, touch=True)
        if hit is not None:
            data = hit.path.read_bytes()
            if data:
                return OutpaintResult(
                    hash=hit.hash, bytes=data, source=hit.source, cache="hit"
                )

        local = await asyncio.to_thread(pad_from_edges, source_bytes)
        if local is None:
            raise RuntimeError("local pad failed")

        source = "local"
        result_bytes = local

        if not has_uniform_edges(source_bytes):
            flux = await self.comfy.outpaint(source_bytes)
            if (
                flux
                and not should_reject_flux_pad(flux, source_bytes)
            ):
                result_bytes = flux
                source = "flux"
            elif flux:
                logger.info("Rejected Flux pad for %s; keeping local", content_hash[:12])

        stored = self.cache.put_by_hash(content_hash, result_bytes, source=source)
        if stored is None:
            raise RuntimeError("cache write failed")
        # Mark flux settled with sidecar (matches native-dash convention).
        if source == "flux":
            (self.cache.cache_dir / f"{content_hash}.flux").touch()

        return OutpaintResult(
            hash=content_hash,
            bytes=result_bytes,
            source=source,
            cache="miss",
        )

    def lookup(self, content_hash: str) -> OutpaintResult | None:
        hit = self.cache.get_by_hash(content_hash, touch=True)
        if hit is None:
            return None
        data = hit.path.read_bytes()
        if not data:
            return None
        return OutpaintResult(hash=hit.hash, bytes=data, source=hit.source, cache="hit")

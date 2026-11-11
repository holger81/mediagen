"""From-scratch generation orchestration: txt2img / img2img / reference images.

Mirrors the outpaint flow (cache hit sync, miss starts a background job with
202/poll) but hashes prompt + style + params + inputs instead of cover+pads.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass

from app.activity import ActivityTracker
from app.cache import MediaCache
from app.comfy_client import ComfyUiOutpaintClient
from app.constants import GENERATE_CACHE_VERSION
from app.image_id import canonicalize_image_bytes

logger = logging.getLogger(__name__)

# Style presets are appended to the free-text prompt; the model is prompt-only,
# so a fragment is all a "style" can be.
STYLE_PRESETS: dict[str, str] = {
    "photoreal": (
        "photorealistic photograph, shot on a DSLR, natural lighting, "
        "fine detail, sharp focus, high resolution"
    ),
    "cinematic": (
        "cinematic film still, dramatic lighting, shallow depth of field, "
        "color graded, anamorphic"
    ),
    "golden-hour": (
        "golden hour sunlight, warm low sun, long soft shadows, atmospheric haze"
    ),
    "watercolor": (
        "watercolor painting, soft washes, visible paper texture, delicate color bleeding"
    ),
    "oil": "oil painting, impasto brush strokes, canvas texture, rich pigments",
    "sketch": "pencil sketch, fine hatching, monochrome, hand-drawn line work",
    "anime": "anime illustration, clean line art, cel shading, vibrant colors",
    "render3d": "3d render, octane render, physically based materials, studio lighting",
    "vintage": "vintage film photograph, faded colors, subtle grain, light leak, 1970s look",
}
DEFAULT_STYLE = "photoreal"


def build_prompt(prompt: str, style: str) -> str:
    """Free text first, then the style preset fragment (if any)."""
    parts = [prompt.strip(), STYLE_PRESETS.get(style, "").strip()]
    return ", ".join(p for p in parts if p)


def generate_hash(
    *,
    prompt: str,
    style: str,
    source_bytes: bytes | None,
    reference_bytes: list[bytes],
    denoise: float,
    width: int,
    height: int,
    seed: int | None,
) -> str:
    """sha256(version || style || prompt || params || canonical image pixels)."""
    digest = hashlib.sha256()
    digest.update(GENERATE_CACHE_VERSION.encode("utf-8"))
    digest.update(b"\0")
    digest.update(style.encode("utf-8"))
    digest.update(b"\0")
    digest.update(prompt.encode("utf-8"))
    digest.update(b"\0")
    seed_tag = str(seed) if seed is not None else "auto"
    digest.update(f"{denoise:.3f}|{width}x{height}|{seed_tag}".encode())
    digest.update(b"\0")
    blobs: list[bytes] = []
    if source_bytes:
        blobs.append(source_bytes)
    blobs.extend(reference_bytes)
    for blob in blobs:
        canonical = canonicalize_image_bytes(blob)
        if canonical is not None:
            digest.update(b"canon4\0")
            digest.update(canonical)
        else:
            digest.update(b"raw\0")
            digest.update(blob)
    return digest.hexdigest()


@dataclass(frozen=True)
class GenerateResult:
    hash: str
    bytes: bytes
    source: str  # gen
    cache: str  # hit | miss


@dataclass(frozen=True)
class GenerateStatus:
    hash: str
    retry_after_s: int
    status: str = "generating"


class GenerateService:
    def __init__(
        self,
        cache: MediaCache,
        comfy: ComfyUiOutpaintClient,
        *,
        retry_after_s: int = 5,
        done_stale_after_s: float = 1800.0,
        activity: ActivityTracker | None = None,
    ) -> None:
        self.cache = cache
        self.comfy = comfy
        self.retry_after_s = max(1, retry_after_s)
        # Failure markers older than this are retried instead of refused forever.
        self.done_stale_after_s = max(60.0, float(done_stale_after_s))
        self.activity = activity or ActivityTracker()
        self._inflight: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

    def is_generating(self, content_hash: str) -> bool:
        task = self._inflight.get(content_hash)
        return task is not None and not task.done()

    def inflight_hashes(self) -> list[str]:
        return [h for h, task in self._inflight.items() if not task.done()]

    def _hit(self, content_hash: str, *, touch: bool = True) -> GenerateResult | None:
        hit = self.cache.get_by_hash(content_hash, touch=touch)
        if hit is None or hit.source != "gen":
            return None
        data = hit.path.read_bytes()
        if not data:
            return None
        return GenerateResult(hash=hit.hash, bytes=data, source="gen", cache="hit")

    async def submit(
        self,
        *,
        prompt: str,
        style: str,
        source_bytes: bytes | None = None,
        reference_bytes: list[bytes] | None = None,
        denoise: float = 1.0,
        width: int = 1024,
        height: int = 1024,
        seed: int | None = None,
    ) -> GenerateResult | GenerateStatus:
        """Cache hit → result. Else start Comfy job → generating."""
        refs = list(reference_bytes or [])
        content_hash = generate_hash(
            prompt=prompt,
            style=style,
            source_bytes=source_bytes,
            reference_bytes=refs,
            denoise=denoise,
            width=width,
            height=height,
            seed=seed,
        )
        hit = self._hit(content_hash)
        if hit is not None:
            return hit
        if self.cache.is_done(content_hash, stale_after_s=self.done_stale_after_s):
            raise RuntimeError("generation failed")

        async with self._lock:
            hit = self._hit(content_hash)
            if hit is not None:
                return hit
            if self.cache.is_done(content_hash, stale_after_s=self.done_stale_after_s):
                raise RuntimeError("generation failed")
            if self.is_generating(content_hash):
                return GenerateStatus(hash=content_hash, retry_after_s=self.retry_after_s)

            task = asyncio.create_task(
                self._run_job(
                    content_hash,
                    prompt=prompt,
                    style=style,
                    source_bytes=source_bytes,
                    reference_bytes=refs,
                    denoise=denoise,
                    width=width,
                    height=height,
                    seed=seed,
                ),
                name=f"generate-{content_hash[:12]}",
            )
            self._inflight[content_hash] = task
            self.activity.start(
                content_hash,
                src_w=width,
                src_h=height,
                pads="",
                out_size=f"{width}x{height}",
                quality_gate=False,
            )
            return GenerateStatus(hash=content_hash, retry_after_s=self.retry_after_s)

    def lookup(self, content_hash: str) -> GenerateResult | GenerateStatus | None:
        """Ready result, generating status, or None if unknown/failed."""
        hit = self._hit(content_hash)
        if hit is not None:
            return hit
        if self.is_generating(content_hash):
            return GenerateStatus(hash=content_hash, retry_after_s=self.retry_after_s)
        return None

    async def _run_job(
        self,
        content_hash: str,
        *,
        prompt: str,
        style: str,
        source_bytes: bytes | None,
        reference_bytes: list[bytes],
        denoise: float,
        width: int,
        height: int,
        seed: int | None,
    ) -> None:
        try:
            image = await self.comfy.generate(
                build_prompt(prompt, style),
                source_bytes=source_bytes,
                reference_bytes=reference_bytes,
                width=width,
                height=height,
                denoise=denoise,
                # Deterministic across cache evictions when the caller gave no seed.
                seed=seed if seed is not None else int(content_hash[:8], 16),
                content_hash=content_hash,
            )
            if not image:
                raise RuntimeError("comfy generation failed")
            stored = self.cache.put_by_hash(content_hash, image, source="gen")
            if stored is None:
                raise RuntimeError("cache write failed")
            self.cache.mark_done(content_hash)
            self.activity.finish(content_hash, phase="done")
        except asyncio.CancelledError:
            self.activity.finish(content_hash, phase="error", error="cancelled")
            raise
        except Exception as exc:
            logger.exception("Generate job failed for %s", content_hash[:12])
            # Prevent infinite Comfy re-queue on repeated identical requests.
            self.cache.mark_done(content_hash)
            self.activity.finish(content_hash, phase="error", error=str(exc))
        finally:
            async with self._lock:
                current = self._inflight.get(content_hash)
                if current is asyncio.current_task():
                    self._inflight.pop(content_hash, None)

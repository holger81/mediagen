"""Outpaint service cache settlement — no infinite Flux re-queue."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path

import pytest
from app.cache import MediaCache
from app.comfy_client import ComfyUiOutpaintClient
from app.outpaint import GeneratingStatus, OutpaintResult, OutpaintService
from PIL import Image, ImageDraw


def _pictorial_png() -> bytes:
    img = Image.new("RGB", (64, 64))
    px = img.load()
    for y in range(64):
        for x in range(64):
            px[x, y] = ((x * 3) % 256, (y * 5) % 256, ((x + y) * 7) % 256)
    draw = ImageDraw.Draw(img)
    draw.ellipse((16, 16, 48, 48), fill=(220, 40, 40))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class CountingComfy(ComfyUiOutpaintClient):
    def __init__(self, workflow_path: Path) -> None:
        super().__init__(base_url="http://127.0.0.1:9", workflow_path=workflow_path)
        self.calls = 0

    async def outpaint(self, source_bytes: bytes) -> bytes | None:
        self.calls += 1
        await asyncio.sleep(0.05)
        return None  # force local fallback


@pytest.mark.asyncio
async def test_pictorial_local_fallback_not_requeried(tmp_path: Path) -> None:
    workflow = Path(__file__).resolve().parents[1] / "workflows" / "album_outpaint_api.json"
    cache = MediaCache(tmp_path, max_items=10)
    comfy = CountingComfy(workflow)
    service = OutpaintService(cache, comfy, retry_after_s=1)
    data = _pictorial_png()

    first = await service.submit(data)
    assert isinstance(first, GeneratingStatus)
    # Wait for background job.
    for _ in range(50):
        await asyncio.sleep(0.05)
        outcome = service.lookup(first.hash)
        if isinstance(outcome, OutpaintResult):
            break
    else:
        pytest.fail("job did not finish")

    assert comfy.calls == 1
    second = await service.submit(data)
    assert isinstance(second, OutpaintResult)
    assert second.cache == "hit"
    assert second.source == "local"
    assert comfy.calls == 1  # must not call Comfy again


@pytest.mark.asyncio
async def test_same_cover_bytes_hit_without_second_comfy(tmp_path: Path) -> None:
    workflow = Path(__file__).resolve().parents[1] / "workflows" / "album_outpaint_api.json"
    cache = MediaCache(tmp_path, max_items=10)
    comfy = CountingComfy(workflow)
    service = OutpaintService(cache, comfy, retry_after_s=1)
    data = _pictorial_png()

    first = await service.submit(data)
    assert isinstance(first, GeneratingStatus)
    for _ in range(50):
        await asyncio.sleep(0.05)
        if isinstance(service.lookup(first.hash), OutpaintResult):
            break

    second = await service.submit(data)
    assert isinstance(second, OutpaintResult)
    assert second.cache == "hit"
    assert comfy.calls == 1

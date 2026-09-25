"""Outpaint service: Flux-only settlement (no local pad fallback)."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path

import pytest
from app.cache import MediaCache
from app.comfy_client import ComfyUiOutpaintClient
from app.layout import OutpaintLayout
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


def _fake_flux_jpeg(source_bytes: bytes) -> bytes:
    pads = OutpaintLayout.defaults()
    src = Image.open(io.BytesIO(source_bytes)).convert("RGB")
    out_w = src.width + pads.pad_left + pads.pad_right
    out_h = src.height + pads.pad_top + pads.pad_bottom
    canvas = Image.new("RGB", (out_w, out_h), src.getpixel((0, 0)))
    canvas.paste(src, (pads.pad_left, pads.pad_top))
    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


class CountingComfy(ComfyUiOutpaintClient):
    def __init__(self, workflow_path: Path, *, succeed: bool = True) -> None:
        super().__init__(base_url="http://127.0.0.1:9", workflow_path=workflow_path)
        self.calls = 0
        self.succeed = succeed

    async def outpaint(self, source_bytes: bytes, *, layout=None) -> bytes | None:
        self.calls += 1
        await asyncio.sleep(0.05)
        if not self.succeed:
            return None
        return _fake_flux_jpeg(source_bytes)


@pytest.fixture(autouse=True)
def _accept_any_flux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.outpaint.accept_flux_pad",
        lambda flux, source, *a, **k: flux if flux else None,
    )


@pytest.mark.asyncio
async def test_flux_success_cached_not_requeried(tmp_path: Path) -> None:
    workflow = Path(__file__).resolve().parents[1] / "workflows" / "album_outpaint_api.json"
    cache = MediaCache(tmp_path, max_items=10)
    comfy = CountingComfy(workflow, succeed=True)
    service = OutpaintService(cache, comfy, retry_after_s=1)
    data = _pictorial_png()

    first = await service.submit(data)
    assert isinstance(first, GeneratingStatus)
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
    assert second.source == "flux"
    assert comfy.calls == 1


@pytest.mark.asyncio
async def test_flux_failure_marks_done_without_jpeg(tmp_path: Path) -> None:
    workflow = Path(__file__).resolve().parents[1] / "workflows" / "album_outpaint_api.json"
    cache = MediaCache(tmp_path, max_items=10)
    comfy = CountingComfy(workflow, succeed=False)
    service = OutpaintService(cache, comfy, retry_after_s=1)
    data = _pictorial_png()

    first = await service.submit(data)
    assert isinstance(first, GeneratingStatus)
    for _ in range(50):
        await asyncio.sleep(0.05)
        if cache.is_done(first.hash) and not service.is_generating(first.hash):
            break
    else:
        pytest.fail("failed job did not settle")

    assert comfy.calls == 1
    assert cache.get_by_hash(first.hash, touch=False) is None
    with pytest.raises(RuntimeError, match="flux outpaint failed"):
        await service.submit(data)
    assert comfy.calls == 1


@pytest.mark.asyncio
async def test_legacy_local_entry_dropped_then_flux(tmp_path: Path) -> None:
    workflow = Path(__file__).resolve().parents[1] / "workflows" / "album_outpaint_api.json"
    cache = MediaCache(tmp_path, max_items=10)
    comfy = CountingComfy(workflow, succeed=True)
    service = OutpaintService(cache, comfy, retry_after_s=1)
    data = _pictorial_png()
    content_hash = cache.hash_for(data)
    assert cache.put_by_hash(content_hash, b"\xff\xd8\xfflocal", source="local") is not None

    first = await service.submit(data)
    assert isinstance(first, GeneratingStatus)
    for _ in range(50):
        await asyncio.sleep(0.05)
        if isinstance(service.lookup(first.hash), OutpaintResult):
            break
    assert comfy.calls == 1
    hit = await service.submit(data)
    assert isinstance(hit, OutpaintResult)
    assert hit.source == "flux"

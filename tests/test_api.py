"""API tests with mocked Comfy (Flux-only)."""

from __future__ import annotations

import asyncio
import io
import time
from pathlib import Path

import pytest
from app.cache import MediaCache
from app.comfy_client import ComfyUiOutpaintClient
from app.config import get_settings
from app.layout import OutpaintLayout
from app.main import app
from app.outpaint import OutpaintService
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw


def _fake_flux_jpeg(source_bytes: bytes, layout: OutpaintLayout | None = None) -> bytes:
    """Build a padded JPEG that looks like a successful Flux result (tests bypass gate)."""
    pads = layout or OutpaintLayout.defaults()
    src = Image.open(io.BytesIO(source_bytes)).convert("RGB")
    out_w = src.width + pads.pad_left + pads.pad_right
    out_h = src.height + pads.pad_top + pads.pad_bottom
    canvas = Image.new("RGB", (out_w, out_h), src.getpixel((0, 0)))
    canvas.paste(src, (pads.pad_left, pads.pad_top))
    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CACHE_MAX_ITEMS", "100")
    monkeypatch.setenv("COMFYUI_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("OUTPAINT_RETRY_AFTER_S", "1")
    get_settings.cache_clear()
    # Tests supply synthetic Flux JPEGs; skip the real quality gate.
    monkeypatch.setattr(
        "app.outpaint.accept_flux_pad",
        lambda flux, source, *a, **k: flux if flux else None,
    )

    class FakeComfy(ComfyUiOutpaintClient):
        async def health(self) -> bool:
            return False

        async def outpaint(self, source_bytes: bytes, *, layout=None) -> bytes | None:
            return _fake_flux_jpeg(source_bytes, layout)

    with TestClient(app) as test_client:
        settings = get_settings()
        cache = MediaCache(settings.cache_dir, max_items=settings.cache_max_items)
        comfy = FakeComfy(
            base_url=settings.comfyui_base_url,
            workflow_path=Path(__file__).resolve().parents[1]
            / "workflows"
            / "album_outpaint_api.json",
        )
        test_client.app.state.settings = settings
        test_client.app.state.cache = cache
        test_client.app.state.comfy = comfy
        test_client.app.state.outpaint = OutpaintService(
            cache, comfy, retry_after_s=settings.outpaint_retry_after_s
        )
        yield test_client
    get_settings.cache_clear()


def _png_bytes(color: tuple[int, int, int] = (12, 34, 56)) -> bytes:
    img = Image.new("RGB", (32, 32), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


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


def _wait_ready(client: TestClient, content_hash: str, *, posts: bytes | None = None):
    ready = None
    for _ in range(60):
        poll = client.get(f"/v1/image/outpaint/{content_hash}")
        if poll.status_code == 200:
            return poll
        if poll.status_code == 404 and posts is not None:
            # Failed job: POST again should 502 once .done is set.
            break
        time.sleep(0.05)
    return ready


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["comfyui"] is False
    assert body["cache_max_items"] == 100


def test_outpaint_returns_202_then_flux(client: TestClient) -> None:
    data = _png_bytes()
    resp = client.post(
        "/v1/image/outpaint",
        files={"image": ("cover.png", data, "image/png")},
    )
    assert resp.status_code == 202
    content_hash = resp.json()["hash"]
    assert resp.headers["X-Outpaint-Pad"] == "256,128,256,128"
    assert resp.headers["X-Outpaint-Size"] == "544x288"

    ready = _wait_ready(client, content_hash)
    assert ready is not None
    assert ready.headers["content-type"].startswith("image/jpeg")
    assert ready.headers["X-Outpaint-Source"] == "flux"
    assert ready.headers["X-Outpaint-Status"] == "ready"

    again = client.post(
        "/v1/image/outpaint",
        files={"image": ("cover.png", data, "image/png")},
    )
    assert again.status_code == 200
    assert again.headers["X-Cache"] == "hit"
    assert again.headers["X-Media-Hash"] == content_hash
    assert again.headers["X-Outpaint-Source"] == "flux"


def test_outpaint_custom_pads(client: TestClient) -> None:
    data = _png_bytes()
    resp = client.post(
        "/v1/image/outpaint",
        files={"image": ("cover.png", data, "image/png")},
        data={
            "pad_left": "40",
            "pad_top": "20",
            "pad_right": "60",
            "pad_bottom": "30",
        },
    )
    assert resp.status_code == 202
    content_hash = resp.json()["hash"]
    assert resp.headers["X-Outpaint-Pad"] == "40,20,60,30"
    ready = _wait_ready(client, content_hash)
    assert ready is not None
    assert ready.headers["X-Outpaint-Size"] == "132x82"
    img = Image.open(io.BytesIO(ready.content))
    assert img.size == (132, 82)


def test_outpaint_canvas_xy(client: TestClient) -> None:
    data = _png_bytes()  # 32x32
    resp = client.post(
        "/v1/image/outpaint",
        files={"image": ("cover.png", data, "image/png")},
        data={
            "out_width": "200",
            "out_height": "100",
            "x": "50",
            "y": "10",
        },
    )
    assert resp.status_code == 202
    assert resp.headers["X-Outpaint-Pad"] == "50,10,118,58"
    ready = _wait_ready(client, resp.json()["hash"])
    assert ready is not None
    assert ready.headers["X-Outpaint-Size"] == "200x100"


def test_outpaint_layout_rejected(client: TestClient) -> None:
    data = _png_bytes()
    resp = client.post(
        "/v1/image/outpaint",
        files={"image": ("cover.png", data, "image/png")},
        data={"pad_left": "10", "out_width": "100"},
    )
    assert resp.status_code == 400


def test_different_layouts_separate_cache(client: TestClient) -> None:
    data = _png_bytes()
    a = client.post(
        "/v1/image/outpaint",
        files={"image": ("cover.png", data, "image/png")},
        data={"pad_left": "10", "pad_top": "10", "pad_right": "10", "pad_bottom": "10"},
    )
    b = client.post(
        "/v1/image/outpaint",
        files={"image": ("cover.png", data, "image/png")},
        data={"pad_left": "20", "pad_top": "10", "pad_right": "10", "pad_bottom": "10"},
    )
    assert a.status_code == 202 and b.status_code == 202
    assert a.json()["hash"] != b.json()["hash"]


def test_outpaint_pictorial_returns_202_then_ready(client: TestClient) -> None:
    class SlowComfy(ComfyUiOutpaintClient):
        async def health(self) -> bool:
            return False

        async def outpaint(self, source_bytes: bytes, *, layout=None) -> bytes | None:
            await asyncio.sleep(0.3)
            return _fake_flux_jpeg(source_bytes, layout)

    settings = get_settings()
    cache: MediaCache = client.app.state.cache
    comfy = SlowComfy(
        base_url=settings.comfyui_base_url,
        workflow_path=Path(__file__).resolve().parents[1] / "workflows" / "album_outpaint_api.json",
    )
    client.app.state.comfy = comfy
    client.app.state.outpaint = OutpaintService(cache, comfy, retry_after_s=1)

    data = _pictorial_png()
    resp = client.post(
        "/v1/image/outpaint",
        files={"image": ("cover.png", data, "image/png")},
    )
    assert resp.status_code == 202
    content_hash = resp.json()["hash"]

    mid = client.get(f"/v1/image/outpaint/{content_hash}")
    assert mid.status_code == 202

    again = client.post(
        "/v1/image/outpaint",
        files={"image": ("cover.png", data, "image/png")},
    )
    assert again.status_code == 202
    assert again.json()["hash"] == content_hash

    ready = _wait_ready(client, content_hash)
    assert ready is not None
    assert ready.headers["X-Outpaint-Source"] == "flux"

    cached = client.post(
        "/v1/image/outpaint",
        files={"image": ("cover.png", data, "image/png")},
    )
    assert cached.status_code == 200
    assert cached.headers["X-Cache"] == "hit"
    assert cached.headers["X-Outpaint-Source"] == "flux"


def test_flux_failure_does_not_store_local(client: TestClient) -> None:
    class FailComfy(ComfyUiOutpaintClient):
        async def health(self) -> bool:
            return False

        async def outpaint(self, source_bytes: bytes, *, layout=None) -> bytes | None:
            await asyncio.sleep(0.05)
            return None

    settings = get_settings()
    cache: MediaCache = client.app.state.cache
    comfy = FailComfy(
        base_url=settings.comfyui_base_url,
        workflow_path=Path(__file__).resolve().parents[1] / "workflows" / "album_outpaint_api.json",
    )
    client.app.state.comfy = comfy
    client.app.state.outpaint = OutpaintService(cache, comfy, retry_after_s=1)

    data = _pictorial_png()
    resp = client.post(
        "/v1/image/outpaint",
        files={"image": ("cover.png", data, "image/png")},
    )
    assert resp.status_code == 202
    content_hash = resp.json()["hash"]

    for _ in range(60):
        poll = client.get(f"/v1/image/outpaint/{content_hash}")
        if poll.status_code == 404 and cache.is_done(content_hash):
            break
        time.sleep(0.05)
    else:
        pytest.fail("failed job did not settle")

    assert cache.get_by_hash(content_hash, touch=False) is None
    failed = client.post(
        "/v1/image/outpaint",
        files={"image": ("cover.png", data, "image/png")},
    )
    assert failed.status_code == 502


def test_legacy_local_cache_is_dropped_for_flux(client: TestClient) -> None:
    data = _pictorial_png()
    cache: MediaCache = client.app.state.cache
    content_hash = cache.hash_for(data)
    assert cache.put_by_hash(content_hash, b"\xff\xd8\xfflocal", source="local") is not None

    resp = client.post(
        "/v1/image/outpaint",
        files={"image": ("cover.png", data, "image/png")},
    )
    assert resp.status_code == 202
    ready = _wait_ready(client, resp.json()["hash"])
    assert ready is not None
    assert ready.headers["X-Outpaint-Source"] == "flux"


def test_lookup_missing(client: TestClient) -> None:
    resp = client.get("/v1/image/outpaint/" + ("a" * 64))
    assert resp.status_code == 404

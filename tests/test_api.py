"""API tests with mocked Comfy."""

from __future__ import annotations

import asyncio
import io
import time
from pathlib import Path

import pytest
from app.cache import MediaCache
from app.comfy_client import ComfyUiOutpaintClient
from app.config import get_settings
from app.main import app
from app.outpaint import OutpaintService
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CACHE_MAX_ITEMS", "100")
    monkeypatch.setenv("COMFYUI_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("OUTPAINT_RETRY_AFTER_S", "1")
    get_settings.cache_clear()

    class FakeComfy(ComfyUiOutpaintClient):
        async def health(self) -> bool:
            return False

        async def outpaint(self, source_bytes: bytes) -> bytes | None:
            return None

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
    """Non-uniform edges so Flux path is taken (async 202)."""
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


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["comfyui"] is False
    assert body["cache_max_items"] == 100


def test_outpaint_uniform_returns_ready_jpeg(client: TestClient) -> None:
    data = _png_bytes()
    resp = client.post(
        "/v1/image/outpaint",
        files={"image": ("cover.png", data, "image/png")},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/jpeg")
    assert resp.headers["X-Cache"] == "miss"
    assert resp.headers["X-Outpaint-Source"] == "local"
    assert resp.headers["X-Outpaint-Status"] == "ready"
    content_hash = resp.headers["X-Media-Hash"]
    assert len(content_hash) == 64

    again = client.post(
        "/v1/image/outpaint",
        files={"image": ("cover.png", data, "image/png")},
    )
    assert again.status_code == 200
    assert again.headers["X-Cache"] == "hit"
    assert again.headers["X-Media-Hash"] == content_hash

    lookup = client.get(f"/v1/image/outpaint/{content_hash}")
    assert lookup.status_code == 200
    assert lookup.headers["X-Cache"] == "hit"


def test_outpaint_pictorial_returns_202_then_ready(client: TestClient) -> None:
    class SlowComfy(ComfyUiOutpaintClient):
        async def health(self) -> bool:
            return False

        async def outpaint(self, source_bytes: bytes) -> bytes | None:
            await asyncio.sleep(0.4)
            return None

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
    body = resp.json()
    assert body["status"] == "generating"
    assert body["retry_after_s"] == 1
    content_hash = body["hash"]
    assert len(content_hash) == 64
    assert resp.headers["Retry-After"] == "1"
    assert resp.headers["X-Outpaint-Status"] == "generating"
    assert resp.headers["X-Media-Hash"] == content_hash

    # Still generating shortly after.
    mid = client.get(f"/v1/image/outpaint/{content_hash}")
    assert mid.status_code == 202
    assert mid.json()["status"] == "generating"

    # Concurrent POST should not start a second job.
    again = client.post(
        "/v1/image/outpaint",
        files={"image": ("cover.png", data, "image/png")},
    )
    assert again.status_code == 202
    assert again.json()["hash"] == content_hash

    ready = None
    for _ in range(40):
        poll = client.get(f"/v1/image/outpaint/{content_hash}")
        if poll.status_code == 200:
            ready = poll
            break
        time.sleep(0.05)
    assert ready is not None
    assert ready.headers["X-Outpaint-Status"] == "ready"
    assert ready.headers["X-Outpaint-Source"] == "local"
    assert ready.headers["content-type"].startswith("image/jpeg")

    # Settled local pad must NOT re-queue Flux on the next play.
    cached = client.post(
        "/v1/image/outpaint",
        files={"image": ("cover.png", data, "image/png")},
    )
    assert cached.status_code == 200
    assert cached.headers["X-Cache"] == "hit"
    assert cached.headers["X-Media-Hash"] == content_hash


def test_pictorial_settled_local_cache_not_requeried(client: TestClient) -> None:
    """A finished local pad for a pictorial cover stays cached (no Comfy spam)."""
    data = _pictorial_png()
    cache: MediaCache = client.app.state.cache
    content_hash = cache.hash_for(data)
    local_jpeg = _png_bytes((90, 90, 90))
    assert cache.put_by_hash(content_hash, local_jpeg, source="local") is not None
    cache.mark_done(content_hash)

    lookup = client.get(f"/v1/image/outpaint/{content_hash}")
    assert lookup.status_code == 200
    assert lookup.headers["X-Outpaint-Source"] == "local"

    resp = client.post(
        "/v1/image/outpaint",
        files={"image": ("cover.png", data, "image/png")},
    )
    assert resp.status_code == 200
    assert resp.headers["X-Cache"] == "hit"
    assert resp.headers["X-Outpaint-Source"] == "local"


def test_lookup_missing(client: TestClient) -> None:
    resp = client.get("/v1/image/outpaint/" + ("a" * 64))
    assert resp.status_code == 404

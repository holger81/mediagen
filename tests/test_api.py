"""API tests with mocked Comfy."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from app.cache import MediaCache
from app.comfy_client import ComfyUiOutpaintClient
from app.config import get_settings
from app.main import app
from app.outpaint import OutpaintService
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CACHE_MAX_ITEMS", "100")
    monkeypatch.setenv("COMFYUI_BASE_URL", "http://127.0.0.1:9")
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
        test_client.app.state.outpaint = OutpaintService(cache, comfy)
        yield test_client
    get_settings.cache_clear()


def _png_bytes(color: tuple[int, int, int] = (12, 34, 56)) -> bytes:
    img = Image.new("RGB", (32, 32), color)
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


def test_outpaint_post_caches_local(client: TestClient) -> None:
    data = _png_bytes()
    resp = client.post(
        "/v1/image/outpaint",
        files={"image": ("cover.png", data, "image/png")},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/jpeg")
    assert resp.headers["X-Cache"] == "miss"
    assert resp.headers["X-Outpaint-Source"] == "local"
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


def test_lookup_missing(client: TestClient) -> None:
    resp = client.get("/v1/image/outpaint/" + ("a" * 64))
    assert resp.status_code == 404

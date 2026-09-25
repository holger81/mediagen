"""Admin UI and cache listing tests."""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import pytest
from app.cache import MediaCache
from app.comfy_client import ComfyUiOutpaintClient
from app.config import get_settings
from app.layout import OutpaintLayout
from app.log_buffer import MemoryLogHandler, install_log_buffer
from app.main import app
from app.outpaint import OutpaintService
from fastapi.testclient import TestClient
from PIL import Image


def _fake_flux_jpeg(source_bytes: bytes, layout=None) -> bytes:
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
    monkeypatch.setenv("ADMIN_TOKEN", "")
    get_settings.cache_clear()
    install_log_buffer(capacity=100)
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


def test_admin_page_and_apis(client: TestClient) -> None:
    data = _png_bytes()
    resp = client.post(
        "/v1/image/outpaint",
        files={"image": ("cover.png", data, "image/png")},
    )
    assert resp.status_code == 202
    content_hash = resp.json()["hash"]
    ready = None
    for _ in range(60):
        poll = client.get(f"/v1/image/outpaint/{content_hash}")
        if poll.status_code == 200:
            ready = poll
            break
        time.sleep(0.05)
    assert ready is not None

    page = client.get("/admin")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert "Cached outpaints" in page.text

    entries = client.get("/admin/api/entries")
    assert entries.status_code == 200
    body = entries.json()
    assert body["total"] >= 1
    assert any(e["hash"] == content_hash for e in body["entries"])

    thumb = client.get(f"/admin/cache/{content_hash}.jpg")
    assert thumb.status_code == 200
    assert thumb.headers["content-type"].startswith("image/jpeg")

    logging.getLogger("app.test").info("admin-log-probe")
    logs = client.get("/admin/api/logs")
    assert logs.status_code == 200
    messages = [line["message"] for line in logs.json()["lines"]]
    assert any("admin-log-probe" in m for m in messages)

    deleted = client.delete(f"/admin/api/entries/{content_hash}")
    assert deleted.status_code == 200
    assert deleted.json()["ok"] is True
    assert client.get(f"/admin/cache/{content_hash}.jpg").status_code == 404
    entries_after = client.get("/admin/api/entries").json()
    assert all(e["hash"] != content_hash for e in entries_after["entries"])


def test_admin_token_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    get_settings.cache_clear()
    with TestClient(app) as test_client:
        settings = get_settings()
        cache = MediaCache(settings.cache_dir, max_items=10)
        comfy = ComfyUiOutpaintClient(
            base_url="http://127.0.0.1:9",
            workflow_path=Path(__file__).resolve().parents[1]
            / "workflows"
            / "album_outpaint_api.json",
        )
        test_client.app.state.settings = settings
        test_client.app.state.cache = cache
        test_client.app.state.comfy = comfy
        test_client.app.state.outpaint = OutpaintService(cache, comfy, retry_after_s=1)
        assert test_client.get("/admin").status_code == 401
        assert test_client.get("/admin?token=wrong").status_code == 401
        ok = test_client.get("/admin?token=secret")
        assert ok.status_code == 200
    get_settings.cache_clear()


def test_memory_log_handler_ring() -> None:
    handler = MemoryLogHandler(capacity=50)
    lg = logging.getLogger("app.ringtest")
    lg.addHandler(handler)
    lg.setLevel(logging.INFO)
    for i in range(5):
        lg.info("line-%s", i)
    snap = handler.snapshot(limit=3)
    assert len(snap) == 3
    assert "line-4" in snap[-1].message
    lg.removeHandler(handler)

"""Generate endpoint + graph tests (Comfy mocked)."""

from __future__ import annotations

import io
import time
from pathlib import Path

import pytest
from app.cache import MediaCache
from app.comfy_client import ComfyUiOutpaintClient, build_generate_graph
from app.config import get_settings
from app.generate import GenerateService, build_prompt, generate_hash
from app.main import app
from fastapi.testclient import TestClient
from PIL import Image


def _png_bytes(size: tuple[int, int] = (64, 48), color=(9, 9, 9)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _fake_gen_jpeg(width: int = 128, height: int = 96) -> bytes:
    img = Image.new("RGB", (width, height), (200, 120, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


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

        async def generate(
            self,
            prompt: str,
            *,
            source_bytes=None,
            reference_bytes=None,
            width=1024,
            height=1024,
            denoise=1.0,
            seed=None,
            content_hash=None,
        ) -> bytes | None:
            return _fake_gen_jpeg(width, height)

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
        test_client.app.state.generate = GenerateService(
            cache, comfy, retry_after_s=settings.outpaint_retry_after_s
        )
        yield test_client
    get_settings.cache_clear()


def _wait_ready(client: TestClient, content_hash: str):
    for _ in range(60):
        poll = client.get(f"/v1/image/generate/{content_hash}")
        if poll.status_code == 200:
            return poll
        if poll.status_code == 404:
            break
        time.sleep(0.05)
    return None


# --- graph wiring -----------------------------------------------------------


def test_graph_txt2img() -> None:
    graph = build_generate_graph("a red barn", width=768, height=512, seed=42)
    assert "17" not in graph and "45" not in graph  # no source path
    assert graph["58"]["class_type"] == "EmptyLatentImage"
    assert graph["58"]["inputs"]["width"] == 768
    sampler = graph["3"]["inputs"]
    assert sampler["latent_image"] == ["58", 0]
    assert sampler["positive"] == ["23", 0]
    assert sampler["negative"] == ["46", 0]
    assert sampler["denoise"] == 1.0
    assert sampler["seed"] == 42
    assert graph["49"]["inputs"]["width"] == 768
    assert graph["49"]["inputs"]["height"] == 512


def test_graph_img2img() -> None:
    graph = build_generate_graph("photoreal", source_name="src.png", denoise=0.6, seed=1)
    assert graph["17"]["inputs"]["image"] == "src.png"
    assert graph["57"]["inputs"]["image"] == ["17", 0]
    assert graph["57"]["inputs"]["crop"] == "disabled"
    assert graph["45"]["inputs"]["pixels"] == ["57", 0]
    sampler = graph["3"]["inputs"]
    assert sampler["latent_image"] == ["45", 0]
    assert sampler["denoise"] == 0.6
    assert "58" not in graph


def test_graph_references() -> None:
    graph = build_generate_graph(
        "derive from these", reference_names=["a.png", "b.png"], seed=7
    )
    ref = graph["48"]["inputs"]
    assert ref["images.image0"] == ["70", 0]
    assert ref["images.image1"] == ["71", 0]
    assert ref["resolution"] == "keep resolution"
    sampler = graph["3"]["inputs"]
    assert sampler["positive"] == ["48", 0]
    assert sampler["negative"] == ["48", 1]
    assert sampler["latent_image"] == ["48", 2]


def test_graph_img2img_with_refs_keeps_source_latent() -> None:
    graph = build_generate_graph(
        "x", source_name="src.png", reference_names=["a.png"], denoise=0.5, seed=2
    )
    assert graph["3"]["inputs"]["latent_image"] == ["45", 0]
    assert graph["3"]["inputs"]["positive"] == ["48", 0]
    assert "58" not in graph


# --- hashing -----------------------------------------------------------------


def _hash(**over) -> str:
    base = dict(
        prompt="a house",
        style="photoreal",
        source_bytes=_png_bytes(),
        reference_bytes=[],
        denoise=0.6,
        width=1024,
        height=1024,
        seed=None,
    )
    base.update(over)
    return generate_hash(**base)


def test_hash_stable_and_sensitive() -> None:
    assert _hash() == _hash()
    assert _hash(prompt="a barn") != _hash()
    assert _hash(style="anime") != _hash()
    assert _hash(denoise=0.5) != _hash()
    assert _hash(width=768) != _hash()
    assert _hash(seed=5) != _hash()
    assert _hash(source_bytes=_png_bytes(color=(1, 2, 3))) != _hash()
    assert _hash(reference_bytes=[_png_bytes()]) != _hash()
    # Container differences (PNG vs BMP of identical pixels) hash the same.
    bmp = io.BytesIO()
    Image.open(io.BytesIO(_png_bytes())).save(bmp, format="BMP")
    assert _hash(source_bytes=bmp.getvalue()) == _hash()


def test_build_prompt() -> None:
    assert build_prompt("a house", "photoreal").startswith("a house, ")
    assert "photorealistic" in build_prompt("a house", "photoreal")
    assert build_prompt("a house", "") == "a house"
    assert build_prompt("", "anime") == build_prompt("", "anime")


# --- API ---------------------------------------------------------------------


def test_generate_txt2img_flow(client: TestClient) -> None:
    resp = client.post("/v1/image/generate", data={"prompt": "a red barn", "style": "anime"})
    assert resp.status_code == 202
    content_hash = resp.json()["hash"]
    assert resp.headers["X-Generate-Status"] == "generating"

    ready = _wait_ready(client, content_hash)
    assert ready is not None
    assert ready.headers["content-type"].startswith("image/jpeg")
    assert ready.headers["X-Generate-Source"] == "gen"
    assert ready.headers["X-Cache"] == "hit"  # lookup after job completion

    again = client.post("/v1/image/generate", data={"prompt": "a red barn", "style": "anime"})
    assert again.status_code == 200
    assert again.headers["X-Cache"] == "hit"
    assert again.headers["X-Media-Hash"] == content_hash


def test_generate_img2img_defaults(client: TestClient) -> None:
    # Source-only restyle: style defaults to photoreal, size follows the source.
    resp = client.post(
        "/v1/image/generate",
        files={"image": ("house.png", _png_bytes((640, 480)), "image/png")},
    )
    assert resp.status_code == 202
    ready = _wait_ready(client, resp.json()["hash"])
    assert ready is not None


def test_generate_with_references(client: TestClient) -> None:
    resp = client.post(
        "/v1/image/generate",
        data={"prompt": "derive from these", "style": "cinematic"},
        files=[
            ("references", ("r0.png", _png_bytes((32, 32)), "image/png")),
            ("references", ("r1.png", _png_bytes((32, 32), (1, 2, 3)), "image/png")),
        ],
    )
    assert resp.status_code == 202
    ready = _wait_ready(client, resp.json()["hash"])
    assert ready is not None


def test_generate_validation(client: TestClient) -> None:
    assert client.post("/v1/image/generate").status_code == 400
    assert (
        client.post("/v1/image/generate", data={"prompt": "x", "style": "nope"}).status_code
        == 400
    )
    assert (
        client.post("/v1/image/generate", data={"prompt": "x", "denoise": "1.5"}).status_code
        == 400
    )
    assert (
        client.post("/v1/image/generate", data={"prompt": "x", "width": "100"}).status_code
        == 400
    )
    assert (
        client.post("/v1/image/generate", data={"prompt": "x", "width": "1023"}).status_code
        == 400
    )
    assert (
        client.post("/v1/image/generate", data={"prompt": "x", "seed": "-3"}).status_code == 400
    )


def test_generate_lookup_unknown(client: TestClient) -> None:
    missing = "0" * 64
    assert client.get(f"/v1/image/generate/{missing}").status_code == 404
    assert client.get("/v1/image/generate/nothex").status_code == 400

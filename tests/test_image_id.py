"""Tests for canonical image identity hashing."""

from __future__ import annotations

import io

from app.cache import MediaCache
from app.image_id import canonicalize_image_bytes, content_hash
from app.layout import OutpaintLayout
from PIL import Image


def _png(color: tuple[int, int, int], size: tuple[int, int] = (32, 32)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _bmp(color: tuple[int, int, int], size: tuple[int, int] = (32, 32)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="BMP")
    return buf.getvalue()


def test_canonical_hash_stable_across_png_and_bmp() -> None:
    a = _png((40, 80, 120))
    b = _bmp((40, 80, 120))
    assert a != b
    assert content_hash(a) == content_hash(b)


def test_different_pixels_different_hash() -> None:
    assert content_hash(_png((0, 0, 0))) != content_hash(_png((255, 0, 0)))


def test_canonicalize_returns_none_for_garbage() -> None:
    assert canonicalize_image_bytes(b"not-an-image") is None
    assert len(content_hash(b"not-an-image")) == 64


def test_cache_hits_same_pixels_different_container(tmp_path) -> None:
    cache = MediaCache(tmp_path, max_items=10)
    src1 = _png((90, 40, 10))
    out = b"\xff\xd8\xffresult"
    cache.put(src1, out, source="flux")
    cache.mark_done(cache.hash_for(src1))

    src2 = _bmp((90, 40, 10))
    assert cache.hash_for(src1) == cache.hash_for(src2)
    hit = cache.get(src2, touch=True)
    assert hit is not None
    assert hit.source == "flux"
    assert hit.path.read_bytes() == out


def test_different_layouts_different_hash() -> None:
    src = _png((10, 20, 30))
    a = content_hash(src, layout=OutpaintLayout(10, 10, 10, 10))
    b = content_hash(src, layout=OutpaintLayout(20, 10, 10, 10))
    assert a != b
    assert content_hash(src) == content_hash(src, layout=OutpaintLayout.defaults())

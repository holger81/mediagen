"""Tests for content-hash cache and two-tier eviction."""

from __future__ import annotations

from pathlib import Path

from app.cache import MediaCache
from app.constants import OUTPAINT_CACHE_VERSION
from app.layout import OutpaintLayout


def test_content_hash_stable(tmp_path: Path) -> None:
    cache = MediaCache(tmp_path, max_items=10)
    a = cache.hash_for(b"hello")
    b = MediaCache.content_hash(b"hello", OUTPAINT_CACHE_VERSION)
    assert a == b
    assert len(a) == 64
    assert cache.hash_for(b"hello!") != a


def test_put_get_increments_hits(tmp_path: Path) -> None:
    cache = MediaCache(tmp_path, max_items=10)
    source = b"cover-bytes"
    jpeg = b"\xff\xd8\xfffakejpeg"
    put = cache.put(source, jpeg, source="local")
    assert put is not None
    assert put.hits == 1
    hit = cache.get(source, touch=True)
    assert hit is not None
    assert hit.hits == 2
    assert hit.path.read_bytes() == jpeg


def test_evict_singles_before_hot(tmp_path: Path) -> None:
    cache = MediaCache(tmp_path, max_items=3)
    # Fill with three one-shot entries.
    for i in range(3):
        cache.put(f"once-{i}".encode(), f"jpeg-{i}".encode(), source="local")
    assert cache.count() == 3

    # Promote first entry to hot (hits >= 2).
    hot_src = b"once-0"
    assert cache.get(hot_src, touch=True) is not None
    hot_hash = cache.hash_for(hot_src)

    # Inserting a 4th entry should drop a probation victim, not the hot one.
    cache.put(b"once-new", b"jpeg-new", source="flux")
    assert cache.count() == 3
    assert cache.get_by_hash(hot_hash, touch=False) is not None

    # Remaining: hot + newest + one other single (oldest singles gone preferentially).
    hashes = {cache.hash_for(f"once-{i}".encode()) for i in range(3)}
    hashes.add(cache.hash_for(b"once-new"))
    present = [h for h in hashes if cache.get_by_hash(h, touch=False) is not None]
    assert hot_hash in present
    assert cache.hash_for(b"once-new") in present


def test_lookup_by_hash(tmp_path: Path) -> None:
    cache = MediaCache(tmp_path, max_items=5)
    src = b"abc"
    cache.put(src, b"jpegdata", source="flux")
    h = cache.hash_for(src)
    hit = cache.get_by_hash(h, touch=False)
    assert hit is not None
    assert hit.source == "flux"


def test_list_entries(tmp_path: Path) -> None:
    cache = MediaCache(tmp_path, max_items=10)
    src = b"cover-a"
    jpeg = b"\xff\xd8\xfffake"
    put = cache.put(src, jpeg, source="flux")
    assert put is not None
    cache.write_layout(put.hash, OutpaintLayout.defaults())
    cache.mark_done(put.hash)
    entries = cache.list_entries()
    assert len(entries) == 1
    assert entries[0].hash == put.hash
    assert entries[0].source == "flux"
    assert entries[0].done is True
    assert entries[0].pads == OutpaintLayout.defaults().header_pad()
    assert entries[0].size_bytes == len(jpeg)

"""Normalize uploads and hash stable image identity (not raw container bytes)."""

from __future__ import annotations

import hashlib
import io
import struct

from PIL import Image

from app.constants import OUTPAINT_CACHE_VERSION


def canonicalize_image_bytes(source_bytes: bytes) -> bytes | None:
    """Decode to RGB pixels for hashing.

    Ignores container differences (PNG vs BMP of the same pixels). Mild JPEG
    re-encodes can still diverge; the settle/``.done`` marker prevents those
    from re-spamming Comfy when the same cover is played again with the same
    bytes (the common playlist case).
    """
    if not source_bytes:
        return None
    try:
        img = Image.open(io.BytesIO(source_bytes))
        img.load()
        rgb = img.convert("RGB")
    except OSError:
        return None
    w, h = rgb.size
    if w < 1 or h < 1:
        return None
    return struct.pack(">II", w, h) + rgb.tobytes()


def content_hash(
    source_bytes: bytes,
    cache_version: str = OUTPAINT_CACHE_VERSION,
) -> str:
    """sha256(version || fingerprint) with fallback to raw bytes."""
    digest = hashlib.sha256()
    digest.update(cache_version.encode("utf-8"))
    canonical = canonicalize_image_bytes(source_bytes)
    if canonical is not None:
        digest.update(b"canon4\0")
        digest.update(canonical)
    else:
        digest.update(b"raw\0")
        digest.update(source_bytes)
    return digest.hexdigest()

"""Tests for local edge pad and quality gate."""

from __future__ import annotations

import io

from app.constants import (
    OUTPAINT_PAD_BOTTOM,
    OUTPAINT_PAD_LEFT,
    OUTPAINT_PAD_RIGHT,
    OUTPAINT_PAD_TOP,
)
from app.local_pad import (
    accept_flux_pad,
    has_uniform_edges,
    pad_from_edges,
    should_reject_flux_pad,
)
from PIL import Image


def _jpeg(color: tuple[int, int, int], size: tuple[int, int] = (64, 64)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def test_pad_from_edges_expands_geometry() -> None:
    src = _jpeg((20, 80, 160), (48, 48))
    out = pad_from_edges(src)
    assert out is not None
    padded = Image.open(io.BytesIO(out))
    assert padded.size == (
        48 + OUTPAINT_PAD_LEFT + OUTPAINT_PAD_RIGHT,
        48 + OUTPAINT_PAD_TOP + OUTPAINT_PAD_BOTTOM,
    )


def test_uniform_black_edges_detected() -> None:
    src = _jpeg((0, 0, 0), (64, 64))
    assert has_uniform_edges(src) is True


def test_pictorial_center_not_uniform() -> None:
    img = Image.new("RGB", (64, 64), (10, 10, 10))
    for y in range(16, 48):
        for x in range(16, 48):
            img.putpixel((x, y), (200, 40, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    # Outer rim is uniform black → still uniform edges.
    assert has_uniform_edges(buf.getvalue()) is True


def test_reject_cream_invent_pad() -> None:
    # Cover: blue edges. Fake "Flux" pad: cream margins.
    cover = _jpeg((30, 90, 180), (40, 40))
    out_w = 40 + OUTPAINT_PAD_LEFT + OUTPAINT_PAD_RIGHT
    out_h = 40 + OUTPAINT_PAD_TOP + OUTPAINT_PAD_BOTTOM
    padded = Image.new("RGB", (out_w, out_h), (230, 220, 200))
    padded.paste(Image.open(io.BytesIO(cover)), (OUTPAINT_PAD_LEFT, OUTPAINT_PAD_TOP))
    buf = io.BytesIO()
    padded.save(buf, format="JPEG", quality=92)
    assert should_reject_flux_pad(buf.getvalue(), cover) is True


def test_accept_matching_edge_continuation() -> None:
    cover = _jpeg((40, 100, 160), (40, 40))
    local = pad_from_edges(cover)
    assert local is not None
    # Local solid pad that matches edges should not look like a Flux invent seam
    # vs itself in the extreme sense — distance is low so should_reject is False.
    assert should_reject_flux_pad(local, cover) is False


def test_feather_pad_seam_rescues_hard_edge() -> None:
    """Hard cover-box seam clears the gate after a short cross-fade."""
    # Matching margins (no invent) plus a thin ring just outside the cover box
    # so only the seam check fails — feather_pad_seam must bring it under gate.
    cover_c = (180, 90, 70)
    ring_c = (90, 45, 35)
    cover = Image.new("RGB", (64, 64), cover_c)
    out_w = 64 + OUTPAINT_PAD_LEFT + OUTPAINT_PAD_RIGHT
    out_h = 64 + OUTPAINT_PAD_TOP + OUTPAINT_PAD_BOTTOM
    canvas = Image.new("RGB", (out_w, out_h), cover_c)
    canvas.paste(cover, (OUTPAINT_PAD_LEFT, OUTPAINT_PAD_TOP))
    px = canvas.load()
    x0, y0 = OUTPAINT_PAD_LEFT, OUTPAINT_PAD_TOP
    x1, y1 = x0 + 64, y0 + 64
    for y in range(y0, y1):
        for d in range(1, 4):
            px[x0 - d, y] = ring_c
            px[x1 + d - 1, y] = ring_c
    for x in range(x0, x1):
        for d in range(1, 4):
            px[x, y0 - d] = ring_c
            px[x, y1 + d - 1] = ring_c
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    padded = buf.getvalue()
    src_buf = io.BytesIO()
    cover.save(src_buf, format="PNG")
    source = src_buf.getvalue()

    assert should_reject_flux_pad(padded, source) is True
    accepted = accept_flux_pad(padded, source)
    assert accepted is not None
    assert should_reject_flux_pad(accepted, source) is False

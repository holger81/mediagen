"""Tests for outpaint layout parse / validate."""

from __future__ import annotations

import pytest
from app.constants import (
    OUTPAINT_MAX_PAD,
    OUTPAINT_PAD_BOTTOM,
    OUTPAINT_PAD_LEFT,
    OUTPAINT_PAD_RIGHT,
    OUTPAINT_PAD_TOP,
)
from app.layout import OutpaintLayout, parse_outpaint_layout


def test_defaults_when_no_fields() -> None:
    layout = parse_outpaint_layout(src_w=64, src_h=64)
    assert layout == OutpaintLayout.defaults()
    assert layout.as_tuple() == (
        OUTPAINT_PAD_LEFT,
        OUTPAINT_PAD_TOP,
        OUTPAINT_PAD_RIGHT,
        OUTPAINT_PAD_BOTTOM,
    )


def test_pads_passthrough() -> None:
    layout = parse_outpaint_layout(
        src_w=100,
        src_h=80,
        pad_left=10,
        pad_top=20,
        pad_right=30,
        pad_bottom=40,
    )
    assert layout.as_tuple() == (10, 20, 30, 40)


def test_canvas_to_pads() -> None:
    # source 100x80 at (50, 25) on 300x200 → pads 50,25,150,95
    layout = parse_outpaint_layout(
        src_w=100,
        src_h=80,
        out_width=300,
        out_height=200,
        x=50,
        y=25,
    )
    assert layout.as_tuple() == (50, 25, 150, 95)


def test_reject_mixed_forms() -> None:
    with pytest.raises(ValueError, match="not both"):
        parse_outpaint_layout(
            src_w=64,
            src_h=64,
            out_width=200,
            out_height=200,
            x=0,
            y=0,
            pad_left=1,
            pad_top=1,
            pad_right=1,
            pad_bottom=1,
        )


def test_reject_partial_pads() -> None:
    with pytest.raises(ValueError, match="incomplete pads"):
        parse_outpaint_layout(src_w=64, src_h=64, pad_left=10)


def test_reject_partial_canvas() -> None:
    with pytest.raises(ValueError, match="incomplete canvas"):
        parse_outpaint_layout(src_w=64, src_h=64, out_width=200, out_height=200)


def test_reject_negative_pad() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        parse_outpaint_layout(
            src_w=64,
            src_h=64,
            pad_left=-1,
            pad_top=0,
            pad_right=0,
            pad_bottom=0,
        )


def test_reject_cover_outside_canvas() -> None:
    with pytest.raises(ValueError, match="does not fit"):
        parse_outpaint_layout(
            src_w=100,
            src_h=100,
            out_width=150,
            out_height=150,
            x=100,
            y=0,
        )


def test_reject_pad_cap() -> None:
    with pytest.raises(ValueError, match="exceeds max"):
        parse_outpaint_layout(
            src_w=64,
            src_h=64,
            pad_left=OUTPAINT_MAX_PAD + 1,
            pad_top=0,
            pad_right=0,
            pad_bottom=0,
        )


def test_explicit_defaults_match_omitted() -> None:
    a = parse_outpaint_layout(src_w=64, src_h=64)
    b = parse_outpaint_layout(
        src_w=64,
        src_h=64,
        pad_left=OUTPAINT_PAD_LEFT,
        pad_top=OUTPAINT_PAD_TOP,
        pad_right=OUTPAINT_PAD_RIGHT,
        pad_bottom=OUTPAINT_PAD_BOTTOM,
    )
    assert a == b

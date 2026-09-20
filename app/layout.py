"""Outpaint canvas/pad layout: parse, validate, normalize to pads."""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image

from app.constants import (
    OUTPAINT_MAX_OUTPUT_SIDE,
    OUTPAINT_MAX_PAD,
    OUTPAINT_MAX_PIXELS,
    OUTPAINT_PAD_BOTTOM,
    OUTPAINT_PAD_LEFT,
    OUTPAINT_PAD_RIGHT,
    OUTPAINT_PAD_TOP,
)


@dataclass(frozen=True)
class OutpaintLayout:
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int

    @classmethod
    def defaults(cls) -> OutpaintLayout:
        return cls(
            pad_left=OUTPAINT_PAD_LEFT,
            pad_top=OUTPAINT_PAD_TOP,
            pad_right=OUTPAINT_PAD_RIGHT,
            pad_bottom=OUTPAINT_PAD_BOTTOM,
        )

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.pad_left, self.pad_top, self.pad_right, self.pad_bottom)

    def header_pad(self) -> str:
        return f"{self.pad_left},{self.pad_top},{self.pad_right},{self.pad_bottom}"

    def header_size(self, src_w: int, src_h: int) -> str:
        return f"{src_w + self.pad_left + self.pad_right}x{src_h + self.pad_top + self.pad_bottom}"

    def layout_tag(self) -> bytes:
        L, T, R, B = self.as_tuple()
        return f"pads:{L},{T},{R},{B}\0".encode()

    def validate_against_source(self, src_w: int, src_h: int) -> None:
        for name, value in (
            ("pad_left", self.pad_left),
            ("pad_top", self.pad_top),
            ("pad_right", self.pad_right),
            ("pad_bottom", self.pad_bottom),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
            if value > OUTPAINT_MAX_PAD:
                raise ValueError(f"{name} exceeds max {OUTPAINT_MAX_PAD}")
        out_w = src_w + self.pad_left + self.pad_right
        out_h = src_h + self.pad_top + self.pad_bottom
        if out_w > OUTPAINT_MAX_OUTPUT_SIDE or out_h > OUTPAINT_MAX_OUTPUT_SIDE:
            raise ValueError(f"output side exceeds max {OUTPAINT_MAX_OUTPUT_SIDE}")
        if out_w * out_h > OUTPAINT_MAX_PIXELS:
            raise ValueError(f"output pixels exceed max {OUTPAINT_MAX_PIXELS}")


_CANVAS_KEYS = ("out_width", "out_height", "x", "y")
_PAD_KEYS = ("pad_left", "pad_top", "pad_right", "pad_bottom")


def source_size(source_bytes: bytes) -> tuple[int, int]:
    try:
        img = Image.open(io.BytesIO(source_bytes))
        img.load()
        w, h = img.size
    except OSError as exc:
        raise ValueError("invalid image") from exc
    if w < 1 or h < 1:
        raise ValueError("invalid image size")
    return w, h


def _parse_optional_int(name: str, raw: str | int | None) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    text = str(raw).strip()
    if text == "":
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def parse_outpaint_layout(
    *,
    src_w: int,
    src_h: int,
    out_width: str | int | None = None,
    out_height: str | int | None = None,
    x: str | int | None = None,
    y: str | int | None = None,
    pad_left: str | int | None = None,
    pad_top: str | int | None = None,
    pad_right: str | int | None = None,
    pad_bottom: str | int | None = None,
) -> OutpaintLayout:
    """Normalize canvas or pad form fields to OutpaintLayout.

    No layout fields → defaults. Any field present → full single family required.
    Mixed canvas + pad families → error.
    """
    canvas = {
        "out_width": _parse_optional_int("out_width", out_width),
        "out_height": _parse_optional_int("out_height", out_height),
        "x": _parse_optional_int("x", x),
        "y": _parse_optional_int("y", y),
    }
    pads = {
        "pad_left": _parse_optional_int("pad_left", pad_left),
        "pad_top": _parse_optional_int("pad_top", pad_top),
        "pad_right": _parse_optional_int("pad_right", pad_right),
        "pad_bottom": _parse_optional_int("pad_bottom", pad_bottom),
    }
    canvas_present = any(v is not None for v in canvas.values())
    pads_present = any(v is not None for v in pads.values())

    if not canvas_present and not pads_present:
        layout = OutpaintLayout.defaults()
        layout.validate_against_source(src_w, src_h)
        return layout

    if canvas_present and pads_present:
        raise ValueError(
            "provide either canvas fields (out_width,out_height,x,y) or pads, not both"
        )

    if pads_present:
        missing = [k for k in _PAD_KEYS if pads[k] is None]
        if missing:
            raise ValueError(f"incomplete pads; missing {', '.join(missing)}")
        pl, pt, pr, pb = pads["pad_left"], pads["pad_top"], pads["pad_right"], pads["pad_bottom"]
        assert pl is not None and pt is not None and pr is not None and pb is not None
        layout = OutpaintLayout(pad_left=pl, pad_top=pt, pad_right=pr, pad_bottom=pb)
        layout.validate_against_source(src_w, src_h)
        return layout

    missing = [k for k in _CANVAS_KEYS if canvas[k] is None]
    if missing:
        raise ValueError(f"incomplete canvas; missing {', '.join(missing)}")
    ow = canvas["out_width"]
    oh = canvas["out_height"]
    ox = canvas["x"]
    oy = canvas["y"]
    assert ow is not None and oh is not None and ox is not None and oy is not None
    if ow < src_w or oh < src_h:
        raise ValueError("output size must be at least the source size")
    if ox < 0 or oy < 0:
        raise ValueError("x and y must be >= 0")
    if ox + src_w > ow or oy + src_h > oh:
        raise ValueError("source does not fit at the given x,y inside the canvas")
    layout = OutpaintLayout(
        pad_left=ox,
        pad_top=oy,
        pad_right=ow - ox - src_w,
        pad_bottom=oh - oy - src_h,
    )
    layout.validate_against_source(src_w, src_h)
    return layout

"""Local edge-pad outpaint + Flux quality gate (Pillow port of AlbumArtLocalOutpaint)."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

from PIL import Image

from app.constants import (
    BLACK_LUMA_MAX,
    JPEG_QUALITY,
    LOCAL_PAD_MAX_STD,
    MAX_BLACK_EDGE_STD,
    MAX_EDGE_STD,
    MAX_PAD_MISMATCH,
    MAX_PAD_MISMATCH_EXTREME,
    MAX_PAD_SEAM,
    OUTPAINT_PAD_BOTTOM,
    OUTPAINT_PAD_LEFT,
    OUTPAINT_PAD_RIGHT,
    OUTPAINT_PAD_TOP,
)


@dataclass(frozen=True)
class SideMeans:
    left: tuple[int, int, int]
    top: tuple[int, int, int]
    right: tuple[int, int, int]
    bottom: tuple[int, int, int]


def _luma(r: float, g: float, b: float) -> float:
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _color_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    dr = a[0] - b[0]
    dg = a[1] - b[1]
    db = a[2] - b[2]
    return math.sqrt(dr * dr + dg * dg + db * db)


def _edge_band(width: int, height: int) -> int:
    return max(2, min(6, min(width, height) // 48))


def _load_rgb(source_bytes: bytes) -> Image.Image | None:
    if not source_bytes:
        return None
    try:
        img = Image.open(io.BytesIO(source_bytes))
        return img.convert("RGB")
    except OSError:
        return None


def _pixels(img: Image.Image) -> list[tuple[int, int, int]]:
    return list(img.getdata())


def _mean_rect(
    pixels: list[tuple[int, int, int]],
    stride: int,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    *,
    snap_black: bool,
) -> tuple[int, int, int]:
    sum_r = sum_g = sum_b = 0.0
    n = 0
    left = max(0, min(stride, x0))
    right = max(0, min(stride, x1))
    top = max(0, y0)
    bottom = max(0, y1)
    height = len(pixels) // stride if stride else 0
    bottom = min(bottom, height)
    for y in range(top, bottom):
        row = y * stride
        for x in range(left, right):
            r, g, b = pixels[row + x]
            sum_r += r
            sum_g += g
            sum_b += b
            n += 1
    if n == 0:
        return (0, 0, 0)
    r = int(round(sum_r / n))
    g = int(round(sum_g / n))
    b = int(round(sum_b / n))
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    if snap_black and _luma(r, g, b) <= BLACK_LUMA_MAX:
        return (0, 0, 0)
    return (r, g, b)


def analyze_side_means(
    width: int,
    height: int,
    pixels: list[tuple[int, int, int]],
    *,
    snap_black: bool = True,
) -> SideMeans:
    band = _edge_band(width, height)
    mean = lambda x0, x1, y0, y1: _mean_rect(  # noqa: E731
        pixels, width, x0, x1, y0, y1, snap_black=snap_black
    )
    return SideMeans(
        left=mean(0, band, 0, height),
        top=mean(0, width, 0, band),
        right=mean(width - band, width, 0, height),
        bottom=mean(0, width, height - band, height),
    )


def analyze_uniform_edge_fill(
    width: int,
    height: int,
    pixels: list[tuple[int, int, int]],
) -> tuple[int, int, int] | None:
    if width < 8 or height < 8 or len(pixels) < width * height:
        return None
    band = _edge_band(width, height)
    samples: list[tuple[int, int, int]] = []

    def add_row(y: int) -> None:
        yy = max(0, min(height - 1, y))
        row = yy * width
        samples.extend(pixels[row : row + width])

    def add_col(x: int) -> None:
        xx = max(0, min(width - 1, x))
        for y in range(band, height - band):
            samples.append(pixels[y * width + xx])

    for i in range(band):
        add_row(i)
        add_row(height - 1 - i)
        add_col(i)
        add_col(width - 1 - i)

    if len(samples) < 16:
        return None

    n = float(len(samples))
    mean_r = sum(c[0] for c in samples) / n
    mean_g = sum(c[1] for c in samples) / n
    mean_b = sum(c[2] for c in samples) / n
    var_acc = 0.0
    for c in samples:
        dr = c[0] - mean_r
        dg = c[1] - mean_g
        db = c[2] - mean_b
        var_acc += dr * dr + dg * dg + db * db
    std = math.sqrt(var_acc / n)
    luminance = _luma(mean_r, mean_g, mean_b)

    if luminance <= BLACK_LUMA_MAX and std <= MAX_BLACK_EDGE_STD:
        return (0, 0, 0)
    if std > MAX_EDGE_STD:
        return None

    fill = (
        max(0, min(255, int(round(mean_r)))),
        max(0, min(255, int(round(mean_g)))),
        max(0, min(255, int(round(mean_b)))),
    )
    if luminance <= BLACK_LUMA_MAX:
        return (0, 0, 0)
    return fill


def has_uniform_edges(source_bytes: bytes) -> bool:
    img = _load_rgb(source_bytes)
    if img is None:
        return False
    return analyze_uniform_edge_fill(img.width, img.height, _pixels(img)) is not None


def pad_from_edges(
    source_bytes: bytes,
    pad_left: int = OUTPAINT_PAD_LEFT,
    pad_top: int = OUTPAINT_PAD_TOP,
    pad_right: int = OUTPAINT_PAD_RIGHT,
    pad_bottom: int = OUTPAINT_PAD_BOTTOM,
) -> bytes | None:
    src = _load_rgb(source_bytes)
    if src is None:
        return None
    w, h = src.size
    if w < 8 or h < 8:
        return None
    pixels = _pixels(src)
    sides = analyze_side_means(w, h, pixels, snap_black=True)
    uniform = analyze_uniform_edge_fill(w, h, pixels)
    out_w = w + pad_left + pad_right
    out_h = h + pad_top + pad_bottom
    out = Image.new("RGB", (out_w, out_h))
    if uniform is not None:
        out.paste(Image.new("RGB", (out_w, out_h), uniform), (0, 0))
    else:
        out.paste(Image.new("RGB", (pad_left, out_h), sides.left), (0, 0))
        out.paste(
            Image.new("RGB", (pad_right, out_h), sides.right), (out_w - pad_right, 0)
        )
        mid_w = out_w - pad_left - pad_right
        out.paste(Image.new("RGB", (mid_w, pad_top), sides.top), (pad_left, 0))
        out.paste(
            Image.new("RGB", (mid_w, pad_bottom), sides.bottom),
            (pad_left, out_h - pad_bottom),
        )
    out.paste(src, (pad_left, pad_top))
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=JPEG_QUALITY)
    data = buf.getvalue()
    return data or None


def pad_mismatch_distance_sides(pad_sides: SideMeans, cover_sides: SideMeans) -> float:
    return max(
        _color_distance(pad_sides.left, cover_sides.left),
        _color_distance(pad_sides.top, cover_sides.top),
        _color_distance(pad_sides.right, cover_sides.right),
        _color_distance(pad_sides.bottom, cover_sides.bottom),
    )


def pad_mismatch_distance(
    padded_bytes: bytes,
    source_bytes: bytes,
    pad_left: int = OUTPAINT_PAD_LEFT,
    pad_top: int = OUTPAINT_PAD_TOP,
    pad_right: int = OUTPAINT_PAD_RIGHT,
    pad_bottom: int = OUTPAINT_PAD_BOTTOM,
) -> float | None:
    padded = _load_rgb(padded_bytes)
    source = _load_rgb(source_bytes)
    if padded is None or source is None:
        return None
    out_w, out_h = padded.size
    if out_w - pad_left - pad_right <= 0 or out_h - pad_top - pad_bottom <= 0:
        return None
    pad_pixels = _pixels(padded)
    src_pixels = _pixels(source)
    pad_sides = SideMeans(
        left=_mean_rect(pad_pixels, out_w, 0, pad_left, 0, out_h, snap_black=False),
        top=_mean_rect(
            pad_pixels, out_w, pad_left, out_w - pad_right, 0, pad_top, snap_black=False
        ),
        right=_mean_rect(
            pad_pixels, out_w, out_w - pad_right, out_w, 0, out_h, snap_black=False
        ),
        bottom=_mean_rect(
            pad_pixels,
            out_w,
            pad_left,
            out_w - pad_right,
            out_h - pad_bottom,
            out_h,
            snap_black=False,
        ),
    )
    cover_sides = analyze_side_means(source.width, source.height, src_pixels, snap_black=False)
    return pad_mismatch_distance_sides(pad_sides, cover_sides)


def pad_seam_distance(
    out_w: int,
    out_h: int,
    pixels: list[tuple[int, int, int]],
    pad_left: int,
    pad_top: int,
    pad_right: int,
    pad_bottom: int,
) -> float:
    cover_w = out_w - pad_left - pad_right
    cover_h = out_h - pad_top - pad_bottom
    if cover_w <= 0 or cover_h <= 0 or len(pixels) < out_w * out_h:
        return 0.0
    band = max(2, min(8, pad_left, pad_top, pad_right, pad_bottom, cover_w // 8, cover_h // 8))
    y0, y1 = pad_top, out_h - pad_bottom
    x0, x1 = pad_left, out_w - pad_right
    mean = lambda a, b, c, d: _mean_rect(pixels, out_w, a, b, c, d, snap_black=False)  # noqa: E731
    return max(
        _color_distance(mean(x0, x0 + band, y0, y1), mean(x0 - band, x0, y0, y1)),
        _color_distance(mean(x1 - band, x1, y0, y1), mean(x1, x1 + band, y0, y1)),
        _color_distance(mean(x0, x1, y0, y0 + band), mean(x0, x1, y0 - band, y0)),
        _color_distance(mean(x0, x1, y1 - band, y1), mean(x0, x1, y1, y1 + band)),
    )


def is_pad_seam_mismatch(
    padded_bytes: bytes,
    pad_left: int = OUTPAINT_PAD_LEFT,
    pad_top: int = OUTPAINT_PAD_TOP,
    pad_right: int = OUTPAINT_PAD_RIGHT,
    pad_bottom: int = OUTPAINT_PAD_BOTTOM,
) -> bool:
    padded = _load_rgb(padded_bytes)
    if padded is None:
        return False
    out_w, out_h = padded.size
    if out_w - pad_left - pad_right <= 0 or out_h - pad_top - pad_bottom <= 0:
        return False
    return (
        pad_seam_distance(out_w, out_h, _pixels(padded), pad_left, pad_top, pad_right, pad_bottom)
        > MAX_PAD_SEAM
    )


def _region_std_dev(
    pixels: list[tuple[int, int, int]],
    stride: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> float:
    sum_r = sum_g = sum_b = 0.0
    n = 0
    for y in range(y0, y1):
        row = y * stride
        for x in range(x0, x1):
            r, g, b = pixels[row + x]
            sum_r += r
            sum_g += g
            sum_b += b
            n += 1
    if n == 0:
        return 0.0
    mean_r, mean_g, mean_b = sum_r / n, sum_g / n, sum_b / n
    var_sum = 0.0
    for y in range(y0, y1):
        row = y * stride
        for x in range(x0, x1):
            r, g, b = pixels[row + x]
            var_sum += (r - mean_r) ** 2 + (g - mean_g) ** 2 + (b - mean_b) ** 2
    return math.sqrt(var_sum / n)


def looks_like_local_solid_pad(
    padded_bytes: bytes,
    pad_left: int = OUTPAINT_PAD_LEFT,
    pad_top: int = OUTPAINT_PAD_TOP,
    pad_right: int = OUTPAINT_PAD_RIGHT,
    pad_bottom: int = OUTPAINT_PAD_BOTTOM,
) -> bool:
    padded = _load_rgb(padded_bytes)
    if padded is None:
        return False
    out_w, out_h = padded.size
    if out_w - pad_left - pad_right <= 0 or out_h - pad_top - pad_bottom <= 0:
        return False
    pixels = _pixels(padded)
    regions = [
        (0, 0, pad_left, out_h),
        (out_w - pad_right, 0, out_w, out_h),
        (pad_left, 0, out_w - pad_right, pad_top),
        (pad_left, out_h - pad_bottom, out_w - pad_right, out_h),
    ]
    return all(
        (x1 - x0) > 0
        and (y1 - y0) > 0
        and _region_std_dev(pixels, out_w, x0, y0, x1, y1) <= LOCAL_PAD_MAX_STD
        for x0, y0, x1, y1 in regions
    )


def feather_pad_seam(
    padded_bytes: bytes,
    pad_left: int = OUTPAINT_PAD_LEFT,
    pad_top: int = OUTPAINT_PAD_TOP,
    pad_right: int = OUTPAINT_PAD_RIGHT,
    pad_bottom: int = OUTPAINT_PAD_BOTTOM,
    *,
    radius: int = 10,
) -> bytes | None:
    """Cross-fade a band straddling the cover box so hard Flux seams blend away.

    Flux Fill often leaves a visible rectangle at the pad boundary even when the
    margin colors are close. A short linear blend across that edge is enough for
    the quality gate and for SoftAtmosphere Crop on the wall.
    """
    padded = _load_rgb(padded_bytes)
    if padded is None:
        return None
    out_w, out_h = padded.size
    cover_w = out_w - pad_left - pad_right
    cover_h = out_h - pad_top - pad_bottom
    if cover_w <= 0 or cover_h <= 0:
        return None
    r = max(2, min(radius, pad_left, pad_top, pad_right, pad_bottom, cover_w // 4, cover_h // 4))
    px = padded.load()
    x0, y0 = pad_left, pad_top
    x1, y1 = out_w - pad_right, out_h - pad_bottom

    def blend(a: tuple[int, ...], b: tuple[int, ...], t: float) -> tuple[int, int, int]:
        t = 0.0 if t < 0 else 1.0 if t > 1 else t
        return (
            int(a[0] + (b[0] - a[0]) * t),
            int(a[1] + (b[1] - a[1]) * t),
            int(a[2] + (b[2] - a[2]) * t),
        )

    # Left / right vertical seams
    for y in range(y0, y1):
        for i in range(1, r + 1):
            t = i / (r + 1)
            # outside ← inside
            if x0 - i >= 0:
                px[x0 - i, y] = blend(px[x0 - i, y], px[min(x0 + i, x1 - 1), y], t)
            if x1 + i - 1 < out_w:
                px[x1 + i - 1, y] = blend(px[x1 + i - 1, y], px[max(x1 - i, x0), y], t)
    # Top / bottom horizontal seams
    for x in range(x0, x1):
        for i in range(1, r + 1):
            t = i / (r + 1)
            if y0 - i >= 0:
                px[x, y0 - i] = blend(px[x, y0 - i], px[x, min(y0 + i, y1 - 1)], t)
            if y1 + i - 1 < out_h:
                px[x, y1 + i - 1] = blend(px[x, y1 + i - 1], px[x, max(y1 - i, y0)], t)

    buf = io.BytesIO()
    padded.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def accept_flux_pad(
    padded_bytes: bytes,
    source_bytes: bytes,
    pad_left: int = OUTPAINT_PAD_LEFT,
    pad_top: int = OUTPAINT_PAD_TOP,
    pad_right: int = OUTPAINT_PAD_RIGHT,
    pad_bottom: int = OUTPAINT_PAD_BOTTOM,
) -> bytes | None:
    """Return JPEG bytes to keep as Flux, or None to fall back to local.

    Tries a short seam feather when the raw pad fails only the hard-edge check.
    """
    if not padded_bytes:
        return None
    src = _load_rgb(source_bytes)
    pad = _load_rgb(padded_bytes)
    if src is None or pad is None:
        return None
    expected = (
        src.size[0] + pad_left + pad_right,
        src.size[1] + pad_top + pad_bottom,
    )
    if pad.size != expected:
        # Wrong geometry — seam metrics would be meaningless.
        return None
    if not should_reject_flux_pad(
        padded_bytes, source_bytes, pad_left, pad_top, pad_right, pad_bottom
    ):
        return padded_bytes
    feathered = feather_pad_seam(
        padded_bytes, pad_left, pad_top, pad_right, pad_bottom, radius=10
    )
    if feathered and not should_reject_flux_pad(
        feathered, source_bytes, pad_left, pad_top, pad_right, pad_bottom
    ):
        return feathered
    return None


def should_reject_flux_pad(
    padded_bytes: bytes,
    source_bytes: bytes,
    pad_left: int = OUTPAINT_PAD_LEFT,
    pad_top: int = OUTPAINT_PAD_TOP,
    pad_right: int = OUTPAINT_PAD_RIGHT,
    pad_bottom: int = OUTPAINT_PAD_BOTTOM,
) -> bool:
    if is_pad_seam_mismatch(padded_bytes, pad_left, pad_top, pad_right, pad_bottom):
        return True
    distance = pad_mismatch_distance(
        padded_bytes, source_bytes, pad_left, pad_top, pad_right, pad_bottom
    )
    if distance is None:
        return False
    if distance <= MAX_PAD_MISMATCH:
        return False
    if distance > MAX_PAD_MISMATCH_EXTREME:
        return True
    return looks_like_local_solid_pad(padded_bytes, pad_left, pad_top, pad_right, pad_bottom)

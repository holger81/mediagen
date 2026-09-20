"""Shared outpaint constants matching ha_native_dash ComfyUiOutpaintClient."""

from __future__ import annotations

OUTPAINT_PAD_LEFT = 256
OUTPAINT_PAD_TOP = 128
OUTPAINT_PAD_RIGHT = 256
OUTPAINT_PAD_BOTTOM = 128

# Caps for requestable layout (translate-only outpaint).
OUTPAINT_MAX_PAD = 2048
OUTPAINT_MAX_OUTPUT_SIDE = 4096
OUTPAINT_MAX_PIXELS = 16_777_216  # 4096^2

# Flux Fill: steer margins away from invented type (empty prompt was inventing glyphs).
OUTPAINT_PROMPT = (
    "seamless photographic background extension matching the cover edges, "
    "soft atmosphere, no text, no letters, no words, no typography, "
    "no logos, no watermark, no signature, no captions"
)
OUTPAINT_NEGATIVE_PROMPT = (
    "text, letters, words, typography, title, caption, logo, watermark, "
    "signature, calligraphy, alphabet, glyphs, writing, font, "
    "illegible text, random characters, brand mark"
)

# Bump when pad strategy / prompt / workflow / hash layout changes.
OUTPAINT_CACHE_VERSION = "no-text-prompt-feather10-layout-v8"

LOAD_IMAGE_NODE_ID = "17"
POSITIVE_PROMPT_NODE_ID = "23"
NEGATIVE_PROMPT_NODE_ID = "46"
WORKFLOW_FILENAME = "album_outpaint_api.json"

# Local pad / quality-gate thresholds (AlbumArtLocalOutpaint).
MAX_EDGE_STD = 36.0
MAX_BLACK_EDGE_STD = 55.0
BLACK_LUMA_MAX = 40.0
MAX_PAD_MISMATCH = 45.0
MAX_PAD_MISMATCH_EXTREME = 85.0
MAX_PAD_SEAM = 38.0
LOCAL_PAD_MAX_STD = 12.0
JPEG_QUALITY = 92

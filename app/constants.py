"""Shared outpaint constants matching ha_native_dash ComfyUiOutpaintClient."""

from __future__ import annotations

OUTPAINT_PAD_LEFT = 256
OUTPAINT_PAD_TOP = 128
OUTPAINT_PAD_RIGHT = 256
OUTPAINT_PAD_BOTTOM = 128

# Intentionally empty — Flux fill extends from edges; instruction prompts hurt quality.
OUTPAINT_PROMPT = ""

# Bump when pad strategy / prompt / workflow quality changes.
OUTPAINT_CACHE_VERSION = "empty-prompt-feather0-v5"

LOAD_IMAGE_NODE_ID = "17"
POSITIVE_PROMPT_NODE_ID = "23"
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

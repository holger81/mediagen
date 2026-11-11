"""Tests for Comfy workflow rewrite helpers."""

from __future__ import annotations

import json
from pathlib import Path

from app.comfy_client import prepare_workflow
from app.constants import (
    NEGATIVE_PROMPT_NODE_ID,
    OUTPAINT_NEGATIVE_PROMPT,
    OUTPAINT_PROMPT,
    POSITIVE_PROMPT_NODE_ID,
)

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = json.loads((ROOT / "workflows" / "album_outpaint_api.json").read_text())


def test_prepare_workflow_sets_image_prompt_seed() -> None:
    prepared = prepare_workflow(WORKFLOW, image_name="uploaded.png", seed=42)
    assert "_meta" not in prepared
    assert prepared["17"]["inputs"]["image"] == "uploaded.png"
    assert prepared[POSITIVE_PROMPT_NODE_ID]["inputs"]["text"] == OUTPAINT_PROMPT
    assert prepared[NEGATIVE_PROMPT_NODE_ID]["class_type"] == "CLIPTextEncode"
    assert prepared[NEGATIVE_PROMPT_NODE_ID]["inputs"]["text"] == OUTPAINT_NEGATIVE_PROMPT
    assert prepared["3"]["inputs"]["seed"] == 42
    # Pads unchanged when layout omitted.
    assert prepared["44"]["inputs"]["left"] == 256
    assert prepared["44"]["inputs"]["top"] == 128
    assert prepared["44"]["inputs"]["feathering"] == 0


def test_prepare_workflow_rewrites_pads() -> None:
    from app.layout import OutpaintLayout

    layout = OutpaintLayout(pad_left=64, pad_top=32, pad_right=96, pad_bottom=48)
    prepared = prepare_workflow(WORKFLOW, image_name="x.png", seed=1, layout=layout)
    assert prepared["44"]["inputs"]["left"] == 64
    assert prepared["44"]["inputs"]["top"] == 32
    assert prepared["44"]["inputs"]["right"] == 96
    assert prepared["44"]["inputs"]["bottom"] == 48


def test_prepare_workflow_injects_no_text_prompts() -> None:
    prepared = prepare_workflow(WORKFLOW, image_name="x.png", seed=1)
    positive = prepared["23"]["inputs"]["text"].lower()
    negative = prepared["46"]["inputs"]["text"].lower()
    assert "no text" in positive
    assert "no border" in positive or "filling the canvas" in positive
    assert "text" in negative
    assert "border" in negative
    assert "polaroid" in negative
    # Negative is a real encode, not ConditioningZeroOut.
    assert prepared["46"]["class_type"] == "CLIPTextEncode"
    assert prepared["38"]["inputs"]["negative"] == ["46", 0]


def test_workflow_template_has_negative_clip_encode() -> None:
    assert WORKFLOW["46"]["class_type"] == "CLIPTextEncode"
    assert WORKFLOW["46"]["inputs"]["clip"] == ["34", 0]


KLEIN_WORKFLOW = json.loads(
    (ROOT / "workflows" / "album_outpaint_klein_api.json").read_text()
)


def test_klein_template_is_lora_reference_graph() -> None:
    """Klein graph must be the AusBoss LoRA recipe, not the old pad/mask graph."""
    assert KLEIN_WORKFLOW["60"]["class_type"] == "LoraLoader"
    assert KLEIN_WORKFLOW["60"]["inputs"]["lora_name"] == (
        "flux-3-image-klein-outpaint.safetensors"
    )
    assert KLEIN_WORKFLOW["60"]["inputs"]["strength_model"] == 1.5
    # Negative side is zeroed conditioning, never a text encode.
    assert KLEIN_WORKFLOW["46"]["class_type"] == "ConditioningZeroOut"
    # KSampler samples a fresh latent from ReferenceMultiLatent (no noise mask,
    # no VAEEncode of a padded cover: denoise 1.0 with a reference canvas).
    assert KLEIN_WORKFLOW["3"]["inputs"]["latent_image"] == ["48", 2]
    assert KLEIN_WORKFLOW["3"]["inputs"]["denoise"] == 1.0
    classes = {node["class_type"] for node in KLEIN_WORKFLOW.values()}
    assert "SetLatentNoiseMask" not in classes
    assert "ImagePadForOutpaint" not in classes
    assert "VAEEncode" not in classes


def test_prepare_workflow_klein_writes_guide_and_pads() -> None:
    from app.constants import KLEIN_OUTPAINT_PROMPT
    from app.layout import OutpaintLayout

    layout = OutpaintLayout(pad_left=162, pad_top=12, pad_right=162, pad_bottom=192)
    prepared = prepare_workflow(
        KLEIN_WORKFLOW,
        image_name="cover.png",
        positive_prompt=KLEIN_OUTPAINT_PROMPT,
        negative_prompt="",
        seed=7,
        layout=layout,
        guide_name="guide.png",
    )
    assert prepared["17"]["inputs"]["image"] == "cover.png"
    assert prepared["54"]["inputs"]["image"] == "guide.png"
    assert prepared["23"]["inputs"]["text"] == KLEIN_OUTPAINT_PROMPT
    # The ConditioningZeroOut negative must survive untouched.
    assert prepared["46"]["class_type"] == "ConditioningZeroOut"
    assert "text" not in prepared["46"]["inputs"]
    assert prepared["3"]["inputs"]["seed"] == 7
    assert prepared["52"]["inputs"]["x"] == 162
    assert prepared["52"]["inputs"]["y"] == 12


def test_build_guide_bytes_gray_canvas() -> None:
    import io

    from app.comfy_client import build_guide_bytes
    from app.layout import OutpaintLayout
    from PIL import Image

    cover = Image.new("RGB", (100, 80), (10, 200, 30))
    buf = io.BytesIO()
    cover.save(buf, format="PNG")
    layout = OutpaintLayout(pad_left=50, pad_top=10, pad_right=50, pad_bottom=30)
    out = Image.open(io.BytesIO(build_guide_bytes(buf.getvalue(), layout)))
    # 100+100=200 -> 192, 80+40=120 -> 112 (multiples of 16).
    assert out.size == (192, 112)
    assert out.getpixel((5, 5)) == (128, 128, 128)  # gray border band
    assert out.getpixel((50 + 50, 10 + 40)) == (10, 200, 30)  # cover at offset

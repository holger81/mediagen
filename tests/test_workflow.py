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
    assert "no text" in prepared["23"]["inputs"]["text"].lower()
    assert "text" in prepared["46"]["inputs"]["text"].lower()
    # Negative is a real encode, not ConditioningZeroOut.
    assert prepared["46"]["class_type"] == "CLIPTextEncode"
    assert prepared["38"]["inputs"]["negative"] == ["46", 0]


def test_workflow_template_has_negative_clip_encode() -> None:
    assert WORKFLOW["46"]["class_type"] == "CLIPTextEncode"
    assert WORKFLOW["46"]["inputs"]["clip"] == ["34", 0]

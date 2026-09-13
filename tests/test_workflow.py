"""Tests for Comfy workflow rewrite helpers."""

from __future__ import annotations

import json
from pathlib import Path

from app.comfy_client import prepare_workflow
from app.constants import OUTPAINT_PROMPT, POSITIVE_PROMPT_NODE_ID

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = json.loads((ROOT / "workflows" / "album_outpaint_api.json").read_text())


def test_prepare_workflow_sets_image_prompt_seed() -> None:
    prepared = prepare_workflow(WORKFLOW, image_name="uploaded.png", seed=42)
    assert "_meta" not in prepared
    assert prepared["17"]["inputs"]["image"] == "uploaded.png"
    assert prepared[POSITIVE_PROMPT_NODE_ID]["inputs"]["text"] == OUTPAINT_PROMPT
    assert prepared["3"]["inputs"]["seed"] == 42
    # Pads unchanged.
    assert prepared["44"]["inputs"]["left"] == 256
    assert prepared["44"]["inputs"]["top"] == 128
    assert prepared["44"]["inputs"]["feathering"] == 0


def test_prepare_workflow_empty_prompt_by_default() -> None:
    prepared = prepare_workflow(WORKFLOW, image_name="x.png", seed=1)
    assert prepared["23"]["inputs"]["text"] == ""

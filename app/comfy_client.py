"""ComfyUI HTTP client for Flux Fill outpainting."""

from __future__ import annotations

import copy
import logging
import random
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from app.constants import (
    LOAD_IMAGE_NODE_ID,
    NEGATIVE_PROMPT_NODE_ID,
    OUTPAINT_NEGATIVE_PROMPT,
    OUTPAINT_PROMPT,
    POSITIVE_PROMPT_NODE_ID,
    WORKFLOW_FILENAME,
)
from app.layout import OutpaintLayout

logger = logging.getLogger(__name__)


def prepare_workflow(
    workflow: dict[str, Any],
    image_name: str,
    positive_prompt: str = OUTPAINT_PROMPT,
    negative_prompt: str = OUTPAINT_NEGATIVE_PROMPT,
    seed: int | None = None,
    *,
    layout: OutpaintLayout | None = None,
) -> dict[str, Any]:
    """Rewrite LoadImage, pads, positive/negative CLIP prompts, and a fresh seed."""
    if seed is None:
        seed = random.randint(0, 2_147_483_647)
    pads = layout if layout is not None else OutpaintLayout.defaults()
    result: dict[str, Any] = {}
    wrote_image = False
    wrote_positive = False
    wrote_negative = False
    for key, value in workflow.items():
        if key == "_meta":
            continue
        if not isinstance(value, dict):
            result[key] = value
            continue
        node = copy.deepcopy(value)
        class_type = node.get("class_type")
        inputs = dict(node.get("inputs") or {})
        if class_type == "LoadImage":
            inputs["image"] = image_name
            node["inputs"] = inputs
            wrote_image = True
        elif class_type == "CLIPTextEncode" and key == POSITIVE_PROMPT_NODE_ID:
            inputs["text"] = positive_prompt
            node["inputs"] = inputs
            wrote_positive = True
        elif class_type == "CLIPTextEncode" and key == NEGATIVE_PROMPT_NODE_ID:
            inputs["text"] = negative_prompt
            node["inputs"] = inputs
            wrote_negative = True
        elif class_type == "KSampler":
            inputs["seed"] = seed
            node["inputs"] = inputs
        elif class_type == "ImagePadForOutpaint":
            inputs["left"] = pads.pad_left
            inputs["top"] = pads.pad_top
            inputs["right"] = pads.pad_right
            inputs["bottom"] = pads.pad_bottom
            node["inputs"] = inputs
        result[key] = node

    if not wrote_image and LOAD_IMAGE_NODE_ID in result:
        node = result[LOAD_IMAGE_NODE_ID]
        if isinstance(node, dict):
            inputs = dict(node.get("inputs") or {})
            inputs["image"] = image_name
            node = {**node, "inputs": inputs}
            result[LOAD_IMAGE_NODE_ID] = node
    if not wrote_positive and POSITIVE_PROMPT_NODE_ID in result:
        node = result[POSITIVE_PROMPT_NODE_ID]
        if isinstance(node, dict):
            inputs = dict(node.get("inputs") or {})
            inputs["text"] = positive_prompt
            node = {**node, "inputs": inputs}
            result[POSITIVE_PROMPT_NODE_ID] = node
    if not wrote_negative and NEGATIVE_PROMPT_NODE_ID in result:
        node = result[NEGATIVE_PROMPT_NODE_ID]
        if isinstance(node, dict):
            inputs = dict(node.get("inputs") or {})
            inputs["text"] = negative_prompt
            # Ensure CLIPTextEncode shape if an old ConditioningZeroOut template sneaks in.
            inputs.setdefault("clip", ["34", 0])
            node = {**node, "class_type": "CLIPTextEncode", "inputs": inputs}
            result[NEGATIVE_PROMPT_NODE_ID] = node
    return result


class ComfyUiOutpaintClient:
    def __init__(
        self,
        base_url: str,
        workflow_path: Path,
        *,
        poll_interval_s: float = 1.5,
        poll_timeout_s: float = 90.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.strip().rstrip("/")
        self.workflow_path = workflow_path
        self.poll_interval_s = poll_interval_s
        self.poll_timeout_s = poll_timeout_s
        self._client = client
        self._owns_client = client is None
        self._workflow_template: dict[str, Any] | None = None

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(15.0, read=60.0, write=60.0, pool=120.0)
            )
        return self._client

    def _load_workflow(self) -> dict[str, Any]:
        if self._workflow_template is None:
            import json

            raw = self.workflow_path.read_text(encoding="utf-8")
            self._workflow_template = json.loads(raw)
        return self._workflow_template

    async def health(self) -> bool:
        if not self.base_url:
            return False
        try:
            resp = await self._http().get(f"{self.base_url}/system_stats")
            return resp.is_success
        except httpx.HTTPError:
            return False

    async def outpaint(
        self,
        source_bytes: bytes,
        *,
        layout: OutpaintLayout | None = None,
    ) -> bytes | None:
        if not self.base_url or not source_bytes:
            return None
        uploaded = await self._upload_image(source_bytes)
        if uploaded is None:
            return None
        prompt_id = await self._queue_prompt(uploaded, layout=layout)
        if prompt_id is None:
            return None
        view = await self._wait_for_output(prompt_id)
        if view is None:
            return None
        return await self._download_view(view)

    async def _upload_image(self, source_bytes: bytes) -> str | None:
        try:
            files = {"image": ("album_cover.png", source_bytes, "application/octet-stream")}
            data = {"type": "input", "overwrite": "true"}
            resp = await self._http().post(f"{self.base_url}/upload/image", files=files, data=data)
            if not resp.is_success:
                logger.warning("Comfy upload failed: %s", resp.status_code)
                return None
            obj = resp.json()
            name = str(obj.get("name") or "").strip()
            subfolder = str(obj.get("subfolder") or "").strip()
            if not name:
                return None
            return f"{subfolder}/{name}" if subfolder else name
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Comfy upload error: %s", exc)
            return None

    async def _queue_prompt(
        self,
        image_name: str,
        *,
        layout: OutpaintLayout | None = None,
    ) -> str | None:
        try:
            workflow = prepare_workflow(self._load_workflow(), image_name=image_name, layout=layout)
            payload = {"prompt": workflow, "client_id": str(uuid.uuid4())}
            resp = await self._http().post(f"{self.base_url}/prompt", json=payload)
            if not resp.is_success:
                logger.warning("Comfy prompt failed: %s", resp.status_code)
                return None
            prompt_id = str(resp.json().get("prompt_id") or "").strip()
            return prompt_id or None
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Comfy prompt error: %s", exc)
            return None

    async def _wait_for_output(self, prompt_id: str) -> dict[str, str] | None:
        import asyncio
        import time

        deadline = time.monotonic() + self.poll_timeout_s
        while time.monotonic() < deadline:
            view = await self._fetch_history_output(prompt_id)
            if view is not None:
                return view
            await asyncio.sleep(self.poll_interval_s)
        logger.warning("Comfy poll timeout for %s", prompt_id)
        return None

    async def _fetch_history_output(self, prompt_id: str) -> dict[str, str] | None:
        try:
            resp = await self._http().get(f"{self.base_url}/history/{prompt_id}")
            if not resp.is_success:
                return None
            root = resp.json()
            entry = root.get(prompt_id) or {}
            outputs = entry.get("outputs") or {}
            for node in outputs.values():
                if not isinstance(node, dict):
                    continue
                images = node.get("images") or []
                for img in images:
                    if not isinstance(img, dict):
                        continue
                    filename = str(img.get("filename") or "").strip()
                    if not filename:
                        continue
                    return {
                        "filename": filename,
                        "subfolder": str(img.get("subfolder") or "").strip(),
                        "type": str(img.get("type") or "output").strip() or "output",
                    }
            return None
        except (httpx.HTTPError, ValueError):
            return None

    async def _download_view(self, view: dict[str, str]) -> bytes | None:
        try:
            # Build query manually to match Comfy's URLEncoder behavior for odd names.
            qs = f"filename={quote(view['filename'])}&type={quote(view['type'])}"
            if view.get("subfolder"):
                qs += f"&subfolder={quote(view['subfolder'])}"
            resp = await self._http().get(f"{self.base_url}/view?{qs}")
            if not resp.is_success:
                return None
            data = resp.content
            return data if data else None
        except httpx.HTTPError as exc:
            logger.warning("Comfy view download error: %s", exc)
            return None


def default_workflow_path(workflows_dir: Path) -> Path:
    return workflows_dir / WORKFLOW_FILENAME

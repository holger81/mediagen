"""ComfyUI HTTP client for Flux Fill outpainting."""

from __future__ import annotations

import copy
import io
import logging
import random
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from PIL import Image

from app.constants import (
    JPEG_QUALITY,
    KLEIN_OUTPAINT_PROMPT,
    LOAD_IMAGE_NODE_ID,
    NEGATIVE_PROMPT_NODE_ID,
    OUTPAINT_NEGATIVE_PROMPT,
    OUTPAINT_PROMPT,
    POSITIVE_PROMPT_NODE_ID,
    WORKFLOW_FILENAME,
)
from app.layout import OutpaintLayout

logger = logging.getLogger(__name__)


# Klein template wires a second LoadImage (node 54) that receives a
# canvas-sized "guide" image used only as reference conditioning.
GUIDE_IMAGE_NODE_ID = "54"

KLEIN_WORKFLOW_FILENAME = "album_outpaint_klein_api.json"

# Klein base checkpoints used for from-scratch generation (no outpaint LoRA).
KLEIN_UNET = "flux3-image-klein-int8-convrot.safetensors"
KLEIN_CLIP = "qwen3vl_4b_int8_convrot.safetensors"
KLEIN_VAE = "flux2-vae.safetensors"


def build_generate_graph(
    prompt: str,
    *,
    source_name: str | None = None,
    reference_names: list[str] | None = None,
    width: int = 1024,
    height: int = 1024,
    denoise: float = 1.0,
    seed: int | None = None,
) -> dict[str, Any]:
    """Build a Klein txt2img / img2img graph with optional reference images.

    img2img (source_name set): LoadImage -> ImageScale -> VAEEncode feeds the
    KSampler latent, and denoise < 1 keeps the structure (walls, windows) while
    texture, materials and lighting get re-rendered. Reference images go through
    ReferenceMultiLatent, which contributes conditioning plus a base latent for
    pure txt2img; with a source image the encoded latent wins and the refs only
    steer style/content. Klein is distilled for CFG 1, so the negative side is
    ConditioningZeroOut (the negative branch is unused anyway).
    """
    if seed is None:
        seed = random.randint(0, 2_147_483_647)
    refs = [name for name in (reference_names or []) if name]
    graph: dict[str, Any] = {
        "31": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": KLEIN_UNET, "weight_dtype": "default"},
        },
        "34": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": KLEIN_CLIP, "type": "flux3", "device": "default"},
        },
        "32": {"class_type": "VAELoader", "inputs": {"vae_name": KLEIN_VAE}},
        "23": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["34", 0]}},
        "46": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["23", 0]}},
        "49": {
            "class_type": "ModelSamplingFlux",
            "inputs": {
                "model": ["31", 0],
                "max_shift": 2.77259,
                "base_shift": 2.77259,
                "width": width,
                "height": height,
            },
        },
    }
    positive: list[Any] = ["23", 0]
    negative: list[Any] = ["46", 0]
    if refs:
        for i, name in enumerate(refs):
            graph[str(70 + i)] = {
                "class_type": "LoadImage",
                "inputs": {"image": name, "upload": "image"},
            }
        ref_inputs: dict[str, Any] = {
            "positive": ["23", 0],
            "negative": ["46", 0],
            "resolution": "keep resolution",
            "vae": ["32", 0],
        }
        for i in range(len(refs)):
            # Dotted autogrow key required by ReferenceMultiLatent.
            ref_inputs[f"images.image{i}"] = [str(70 + i), 0]
        graph["48"] = {"class_type": "ReferenceMultiLatent", "inputs": ref_inputs}
        positive = ["48", 0]
        negative = ["48", 1]
    if source_name:
        graph["17"] = {
            "class_type": "LoadImage",
            "inputs": {"image": source_name, "upload": "image"},
        }
        graph["57"] = {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["17", 0],
                "upscale_method": "lanczos",
                "width": width,
                "height": height,
                "crop": "disabled",
            },
        }
        graph["45"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["57", 0], "vae": ["32", 0]}}
        latent: list[Any] = ["45", 0]
    else:
        graph["58"] = {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        }
        latent = ["48", 2] if refs else ["58", 0]
    graph["3"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["49", 0],
            "seed": seed,
            "steps": 16,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": denoise,
            "positive": positive,
            "negative": negative,
            "latent_image": latent,
        },
    }
    graph["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["32", 0]}}
    graph["9"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["8", 0], "filename_prefix": "mediagen_gen"},
    }
    return graph


def build_guide_bytes(source_bytes: bytes, layout: OutpaintLayout) -> bytes | None:
    """Compose the canvas-sized reference guide for the Klein outpaint LoRA.

    Flat gray #808080 canvas with the untouched cover pasted at the pad offset
    (1:1 scale). This is the exact recipe the AusBoss FLUX 3 Klein outpaint LoRA
    was trained on: the model is told to replace the solid gray with a scene
    continuation, so anything richer (edge replication, blur, scaled copies)
    makes it repaint the border as an enlarged duplicate of the cover. Canvas
    size matches what the graph renders: cover + pads truncated to multiples
    of 16 by the Flux VAE.
    """
    try:
        img = Image.open(io.BytesIO(source_bytes))
        img.load()
        img = img.convert("RGB")
    except OSError:
        return None
    w, h = img.size
    pad_l, pad_t, pad_r, pad_b = layout.as_tuple()
    cw = (w + pad_l + pad_r) // 16 * 16
    ch = (h + pad_t + pad_b) // 16 * 16
    if cw < 16 or ch < 16:
        return None
    canvas = Image.new("RGB", (cw, ch), (128, 128, 128))
    canvas.paste(img, (pad_l, pad_t))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def fit_to_canvas_bytes(
    image_bytes: bytes, layout: OutpaintLayout, src_w: int, src_h: int
) -> bytes:
    """Stretch Flux output back to the exact requested canvas size.

    The Flux VAE renders the canvas truncated to multiples of 16 (520 -> 512);
    clients that compare the returned image against their widget geometry
    (ha_native_dash) reject anything but an exact match. Only correct plain
    truncation (<=16 px per axis); anything further off is served unchanged.
    """
    target_w = src_w + layout.pad_left + layout.pad_right
    target_h = src_h + layout.pad_top + layout.pad_bottom
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except (OSError, ValueError):
        return image_bytes
    if img.size == (target_w, target_h):
        img.close()
        return image_bytes
    if abs(target_w - img.width) > 16 or abs(target_h - img.height) > 16:
        logger.warning(
            "flux output %dx%d too far from canvas %dx%d; serving as-is",
            img.width,
            img.height,
            target_w,
            target_h,
        )
        img.close()
        return image_bytes
    resized = img.convert("RGB").resize((target_w, target_h), Image.LANCZOS)
    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=JPEG_QUALITY)
    img.close()
    resized.close()
    return buf.getvalue()


def _workflow_seed(workflow: dict[str, Any]) -> int | None:
    for node in workflow.values():
        if isinstance(node, dict) and node.get("class_type") == "KSampler":
            seed = (node.get("inputs") or {}).get("seed")
            return int(seed) if seed is not None else None
    return None


def prepare_workflow(
    workflow: dict[str, Any],
    image_name: str,
    positive_prompt: str = OUTPAINT_PROMPT,
    negative_prompt: str = OUTPAINT_NEGATIVE_PROMPT,
    seed: int | None = None,
    *,
    layout: OutpaintLayout | None = None,
    guide_name: str | None = None,
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
            if key == GUIDE_IMAGE_NODE_ID:
                # Klein reference-guide slot; keep template name when no guide built.
                if guide_name:
                    inputs["image"] = guide_name
                    node["inputs"] = inputs
            else:
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
        elif class_type == "ImageCompositeMasked":
            # Klein keep-center paste-back: the untouched cover goes back at
            # the pad offset so it lands exactly where the canvas expects it.
            inputs["x"] = pads.pad_left
            inputs["y"] = pads.pad_top
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
        # Only rewrite real prompt nodes; the Klein template keeps node 46 as
        # ConditioningZeroOut (LoRA recipe), which must stay untouched.
        if isinstance(node, dict) and node.get("class_type") == "CLIPTextEncode":
            inputs = dict(node.get("inputs") or {})
            inputs["text"] = negative_prompt
            node = {**node, "inputs": inputs}
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
        activity: Any | None = None,
    ) -> None:
        self.base_url = base_url.strip().rstrip("/")
        self.workflow_path = workflow_path
        self.poll_interval_s = poll_interval_s
        self.poll_timeout_s = poll_timeout_s
        self._client = client
        self._owns_client = client is None
        self._workflow_template: dict[str, Any] | None = None
        self.activity = activity
        # Klein + LoRA graphs use the trigger prompt and a zeroed negative.
        self._is_klein = workflow_path.name == KLEIN_WORKFLOW_FILENAME

    def _note(self, content_hash: str | None, **kwargs: Any) -> None:
        if self.activity is None or not content_hash:
            return
        self.activity.update(content_hash, **kwargs)

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
        content_hash: str | None = None,
    ) -> bytes | None:
        if not self.base_url or not source_bytes:
            return None
        # Unique per-cover names: concurrent requests must not overwrite each
        # other's upload while a render is still reading the file.
        tag = (content_hash or uuid.uuid4().hex)[:12]
        self._note(content_hash, phase="uploading")
        uploaded = await self._upload_image(source_bytes, f"album_cover_{tag}.png")
        if uploaded is None:
            self._note(content_hash, phase="error", error="comfy upload failed")
            return None
        guide_name = await self._maybe_upload_guide(source_bytes, layout, uploaded, tag)
        self._note(content_hash, phase="prompt", comfy_image=uploaded)
        prompt_id = await self._queue_prompt(
            uploaded, layout=layout, content_hash=content_hash, guide_name=guide_name
        )
        if prompt_id is None:
            self._note(content_hash, phase="error", error="comfy prompt failed")
            return None
        self._note(content_hash, phase="waiting", prompt_id=prompt_id)
        view = await self._wait_for_output(prompt_id)
        if view is None:
            self._note(content_hash, phase="error", error="comfy poll timeout")
            return None
        self._note(
            content_hash,
            phase="downloading",
            output_filename=view.get("filename"),
        )
        return await self._download_view(view)

    async def generate(
        self,
        prompt: str,
        *,
        source_bytes: bytes | None = None,
        reference_bytes: list[bytes] | None = None,
        width: int = 1024,
        height: int = 1024,
        denoise: float = 1.0,
        seed: int | None = None,
        content_hash: str | None = None,
    ) -> bytes | None:
        """txt2img / img2img render on the Klein base graph (no outpaint LoRA)."""
        if not self.base_url:
            return None
        tag = (content_hash or uuid.uuid4().hex)[:12]
        self._note(content_hash, phase="uploading")
        source_name: str | None = None
        if source_bytes:
            source_name = await self._upload_image(source_bytes, f"gen_source_{tag}.png")
            if source_name is None:
                self._note(content_hash, phase="error", error="comfy source upload failed")
                return None
        ref_names: list[str] = []
        for i, blob in enumerate(reference_bytes or []):
            name = await self._upload_image(blob, f"gen_ref{i}_{tag}.png")
            if name is None:
                self._note(content_hash, phase="error", error="comfy reference upload failed")
                return None
            ref_names.append(name)
        graph = build_generate_graph(
            prompt,
            source_name=source_name,
            reference_names=ref_names,
            width=width,
            height=height,
            denoise=denoise,
            seed=seed,
        )
        self._note(
            content_hash,
            phase="prompt",
            seed=graph["3"]["inputs"]["seed"],
            out_size=f"{width}x{height}",
            denoise=denoise,
            refs=len(ref_names),
        )
        prompt_id = await self._queue_graph(graph)
        if prompt_id is None:
            self._note(content_hash, phase="error", error="comfy prompt failed")
            return None
        self._note(content_hash, phase="waiting", prompt_id=prompt_id)
        view = await self._wait_for_output(prompt_id)
        if view is None:
            self._note(content_hash, phase="error", error="comfy poll timeout")
            return None
        self._note(content_hash, phase="downloading", output_filename=view.get("filename"))
        return await self._download_view(view)

    async def _queue_graph(self, graph: dict[str, Any]) -> str | None:
        """Queue a programmatically built graph (generate path)."""
        try:
            client_id = str(uuid.uuid4())
            resp = await self._http().post(
                f"{self.base_url}/prompt", json={"prompt": graph, "client_id": client_id}
            )
            if not resp.is_success:
                logger.warning("Comfy prompt failed: %s %s", resp.status_code, resp.text[:400])
                return None
            prompt_id = str(resp.json().get("prompt_id") or "").strip()
            return prompt_id or None
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Comfy prompt error: %s", exc)
            return None

    async def _maybe_upload_guide(
        self,
        source_bytes: bytes,
        layout: OutpaintLayout | None,
        fallback_name: str,
        tag: str,
    ) -> str | None:
        """Build and upload the canvas-sized reference guide for Klein graphs.

        Returns the guide image name, or the plain cover name when the template
        has no guide slot or the guide could not be built (older Fill graphs).
        """
        if GUIDE_IMAGE_NODE_ID not in self._load_workflow():
            return None
        pads = layout if layout is not None else OutpaintLayout.defaults()
        guide_bytes = build_guide_bytes(source_bytes, pads)
        if guide_bytes is None:
            logger.warning("guide build failed, falling back to cover reference")
            return fallback_name
        name = await self._upload_image(guide_bytes, f"album_guide_{tag}.png")
        if name is None:
            logger.warning("guide upload failed, falling back to cover reference")
            return fallback_name
        return name

    async def _upload_image(
        self, source_bytes: bytes, filename: str = "album_cover.png"
    ) -> str | None:
        try:
            files = {"image": (filename, source_bytes, "application/octet-stream")}
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
        content_hash: str | None = None,
        guide_name: str | None = None,
    ) -> str | None:
        try:
            client_id = str(uuid.uuid4())
            if self._is_klein:
                positive_prompt = KLEIN_OUTPAINT_PROMPT
                # Negative side is ConditioningZeroOut in the Klein graph.
                negative_prompt = ""
            else:
                positive_prompt = OUTPAINT_PROMPT
                negative_prompt = OUTPAINT_NEGATIVE_PROMPT
            workflow = prepare_workflow(
                self._load_workflow(),
                image_name=image_name,
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                layout=layout,
                guide_name=guide_name,
            )
            pads = layout if layout is not None else OutpaintLayout.defaults()
            self._note(
                content_hash,
                comfy_image=image_name,
                seed=_workflow_seed(workflow),
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                client_id=client_id,
                pads=pads.header_pad(),
            )
            payload = {"prompt": workflow, "client_id": client_id}
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


def default_workflow_path(workflows_dir: Path, model: str = "flux") -> Path:
    if model == "klein":
        return workflows_dir / KLEIN_WORKFLOW_FILENAME
    return workflows_dir / WORKFLOW_FILENAME

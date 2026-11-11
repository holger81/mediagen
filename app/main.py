"""FastAPI entrypoint for mediagen."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from app.activity import ActivityTracker
from app.admin import router as admin_router
from app.cache import MediaCache
from app.comfy_client import ComfyUiOutpaintClient, default_workflow_path
from app.config import Settings, get_settings
from app.constants import OUTPAINT_CACHE_VERSION
from app.generate import (
    DEFAULT_STYLE,
    STYLE_PRESETS,
    GenerateResult,
    GenerateService,
    GenerateStatus,
)
from app.layout import parse_outpaint_layout, source_size
from app.log_buffer import install_log_buffer
from app.outpaint import GeneratingStatus, OutpaintResult, OutpaintService
from app.runtime_settings import apply_runtime_overrides, load_runtime_settings


def _workflows_dir(settings: Settings) -> Path:
    path = settings.workflows_dir
    if path.is_dir():
        return path
    # Dev / pytest: fall back to repo workflows/
    repo = Path(__file__).resolve().parent.parent / "workflows"
    return repo if repo.is_dir() else path


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    install_log_buffer(capacity=settings.admin_log_capacity)
    # Outpaint model switch (flux | klein) via persisted runtime setting.
    # cache_version must stay exactly OUTPAINT_CACHE_VERSION for every model:
    # the dashboard computes expected hashes client-side with that string, so a
    # model-specific suffix makes its GETs miss and re-POST (duplicate renders).
    outpaint_model = str(
        load_runtime_settings(settings.cache_dir).get("outpaint_model", "flux")
    ).lower()
    cache_version = OUTPAINT_CACHE_VERSION
    cache = MediaCache(
        settings.cache_dir, max_items=settings.cache_max_items, cache_version=cache_version
    )
    activity = ActivityTracker()
    comfy = ComfyUiOutpaintClient(
        base_url=settings.comfyui_base_url,
        workflow_path=default_workflow_path(_workflows_dir(settings), model=outpaint_model),
        poll_interval_s=settings.outpaint_poll_interval_s,
        poll_timeout_s=settings.outpaint_poll_timeout_s,
        activity=activity,
    )
    app.state.settings = settings
    app.state.cache = cache
    app.state.comfy = comfy
    app.state.activity = activity
    gate = apply_runtime_overrides(settings.cache_dir, flux_quality_gate=settings.flux_quality_gate)
    app.state.outpaint = OutpaintService(
        cache,
        comfy,
        retry_after_s=settings.outpaint_retry_after_s,
        flux_quality_gate=gate,
        activity=activity,
    )
    # From-scratch generation (txt2img/img2img/refs) builds its graph in code,
    # so it reuses the same Comfy client (workflow template is outpaint-only).
    app.state.generate = GenerateService(
        cache,
        comfy,
        retry_after_s=settings.outpaint_retry_after_s,
        activity=activity,
    )
    try:
        yield
    finally:
        await comfy.aclose()


app = FastAPI(
    title="mediagen",
    version="0.1.0",
    description="Media generation API (image outpaint first)",
    lifespan=lifespan,
)

_settings = get_settings()
_origins = [o.strip() for o in _settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(admin_router)


def _layout_headers(layout_result: OutpaintResult | GeneratingStatus) -> dict[str, str]:
    return {
        "X-Outpaint-Pad": layout_result.layout.header_pad(),
        "X-Outpaint-Size": layout_result.layout.header_size(
            layout_result.src_w, layout_result.src_h
        ),
    }


def _jpeg_response(result: OutpaintResult) -> Response:
    return Response(
        content=result.bytes,
        media_type="image/jpeg",
        headers={
            "X-Media-Hash": result.hash,
            "X-Cache": result.cache,
            "X-Outpaint-Source": result.source,
            "X-Outpaint-Status": "ready",
            **_layout_headers(result),
        },
    )


def _generating_response(status: GeneratingStatus) -> JSONResponse:
    return JSONResponse(
        status_code=202,
        content={
            "status": status.status,
            "hash": status.hash,
            "retry_after_s": status.retry_after_s,
        },
        headers={
            "Retry-After": str(status.retry_after_s),
            "X-Media-Hash": status.hash,
            "X-Outpaint-Status": "generating",
            "X-Cache": "miss",
            **_layout_headers(status),
        },
    )


@app.get("/health")
async def health(request: Request) -> JSONResponse:
    settings: Settings = request.app.state.settings
    comfy: ComfyUiOutpaintClient = request.app.state.comfy
    cache: MediaCache = request.app.state.cache
    comfy_ok = await comfy.health()
    return JSONResponse(
        {
            "status": "ok" if comfy_ok else "degraded",
            "comfyui": comfy_ok,
            "comfyui_base_url": settings.comfyui_base_url,
            "cache_dir": str(settings.cache_dir),
            "cache_items": cache.count(),
            "cache_max_items": settings.cache_max_items,
        }
    )


@app.post("/v1/image/outpaint")
async def image_outpaint(
    request: Request,
    image: Annotated[UploadFile, File(...)],
    out_width: Annotated[str | None, Form()] = None,
    out_height: Annotated[str | None, Form()] = None,
    x: Annotated[str | None, Form()] = None,
    y: Annotated[str | None, Form()] = None,
    pad_left: Annotated[str | None, Form()] = None,
    pad_top: Annotated[str | None, Form()] = None,
    pad_right: Annotated[str | None, Form()] = None,
    pad_bottom: Annotated[str | None, Form()] = None,
) -> Response:
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty image")
    try:
        src_w, src_h = source_size(data)
        layout = parse_outpaint_layout(
            src_w=src_w,
            src_h=src_h,
            out_width=out_width,
            out_height=out_height,
            x=x,
            y=y,
            pad_left=pad_left,
            pad_top=pad_top,
            pad_right=pad_right,
            pad_bottom=pad_bottom,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    service: OutpaintService = request.app.state.outpaint
    try:
        outcome = await service.submit(data, layout=layout, src_w=src_w, src_h=src_h)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if isinstance(outcome, GeneratingStatus):
        return _generating_response(outcome)
    return _jpeg_response(outcome)


@app.get("/v1/image/outpaint/{sha256}")
async def image_outpaint_lookup(sha256: str, request: Request) -> Response:
    if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256.lower()):
        raise HTTPException(status_code=400, detail="invalid sha256")
    service: OutpaintService = request.app.state.outpaint
    outcome = service.lookup(sha256.lower())
    if outcome is None:
        raise HTTPException(status_code=404, detail="not cached")
    if isinstance(outcome, GeneratingStatus):
        return _generating_response(outcome)
    return _jpeg_response(outcome)


def _gen_ready_response(result: GenerateResult) -> Response:
    return Response(
        content=result.bytes,
        media_type="image/jpeg",
        headers={
            "X-Media-Hash": result.hash,
            "X-Cache": result.cache,
            "X-Generate-Source": result.source,
            "X-Generate-Status": "ready",
        },
    )


def _gen_generating_response(status: GenerateStatus) -> JSONResponse:
    return JSONResponse(
        status_code=202,
        content={
            "status": status.status,
            "hash": status.hash,
            "retry_after_s": status.retry_after_s,
        },
        headers={
            "Retry-After": str(status.retry_after_s),
            "X-Media-Hash": status.hash,
            "X-Cache": "miss",
            "X-Generate-Status": "generating",
        },
    )


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())


def _parse_dim(value: str | None, name: str, default: int) -> int:
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid {name}") from exc
    if not 256 <= parsed <= 2048 or parsed % 16:
        raise HTTPException(
            status_code=400, detail=f"{name} must be a multiple of 16 between 256 and 2048"
        )
    return parsed


def _round_dim(value: int) -> int:
    """Snap a source dimension to a Flux-friendly multiple of 16."""
    return max(256, min(2048, max(16, value // 16 * 16)))


@app.post("/v1/image/generate")
async def image_generate(
    request: Request,
    prompt: Annotated[str | None, Form()] = None,
    style: Annotated[str | None, Form()] = None,
    image: Annotated[UploadFile | None, File()] = None,
    references: Annotated[list[UploadFile] | None, File()] = None,
    denoise: Annotated[str | None, Form()] = None,
    width: Annotated[str | None, Form()] = None,
    height: Annotated[str | None, Form()] = None,
    seed: Annotated[str | None, Form()] = None,
) -> Response:
    """txt2img (prompt only) / img2img (source image, denoise < 1) with optional
    reference images. Same 202-then-poll contract as the outpaint endpoint."""
    prompt = (prompt or "").strip()
    style = (style or "").strip().lower()
    if style and style not in STYLE_PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown style (choose: {', '.join(sorted(STYLE_PRESETS))})",
        )
    source_bytes = await image.read() if image is not None else b""
    if image is not None and not source_bytes:
        raise HTTPException(status_code=400, detail="empty image")
    ref_blobs: list[bytes] = []
    for ref in references or []:
        blob = await ref.read()
        if blob:
            ref_blobs.append(blob)
    if not style:
        if prompt or ref_blobs:
            style = ""  # free text only (refs alone still steer the sampler)
        elif source_bytes:
            style = DEFAULT_STYLE  # restyle request without an explicit style
        else:
            raise HTTPException(
                status_code=400, detail="provide a prompt, style, image, or reference"
            )

    src_w = src_h = 0
    if source_bytes:
        try:
            src_w, src_h = source_size(source_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        denoise_val = float(denoise) if denoise not in (None, "") else (0.6 if src_w else 1.0)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid denoise") from exc
    if not 0.0 < denoise_val <= 1.0:
        raise HTTPException(status_code=400, detail="denoise must be in (0, 1]")
    seed_val: int | None = None
    if seed not in (None, ""):
        try:
            seed_val = int(seed)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid seed") from exc
        if not 0 <= seed_val <= 2_147_483_647:
            raise HTTPException(status_code=400, detail="seed out of range")
    # Default output size: keep the source photo's aspect, else square 1024.
    width_val = _parse_dim(width, "width", _round_dim(src_w) if src_w else 1024)
    height_val = _parse_dim(height, "height", _round_dim(src_h) if src_h else 1024)

    service: GenerateService = request.app.state.generate
    try:
        outcome = await service.submit(
            prompt=prompt,
            style=style,
            source_bytes=source_bytes or None,
            reference_bytes=ref_blobs,
            denoise=denoise_val,
            width=width_val,
            height=height_val,
            seed=seed_val,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if isinstance(outcome, GenerateStatus):
        return _gen_generating_response(outcome)
    return _gen_ready_response(outcome)


@app.get("/v1/image/generate/{sha256}")
async def image_generate_lookup(sha256: str, request: Request) -> Response:
    if not _is_sha256(sha256):
        raise HTTPException(status_code=400, detail="invalid sha256")
    service: GenerateService = request.app.state.generate
    outcome = service.lookup(sha256.lower())
    if outcome is None:
        raise HTTPException(status_code=404, detail="not cached")
    if isinstance(outcome, GenerateStatus):
        return _gen_generating_response(outcome)
    return _gen_ready_response(outcome)

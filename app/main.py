"""FastAPI entrypoint for mediagen."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from app.cache import MediaCache
from app.comfy_client import ComfyUiOutpaintClient, default_workflow_path
from app.config import Settings, get_settings
from app.layout import parse_outpaint_layout, source_size
from app.outpaint import GeneratingStatus, OutpaintResult, OutpaintService


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
    cache = MediaCache(settings.cache_dir, max_items=settings.cache_max_items)
    comfy = ComfyUiOutpaintClient(
        base_url=settings.comfyui_base_url,
        workflow_path=default_workflow_path(_workflows_dir(settings)),
        poll_interval_s=settings.outpaint_poll_interval_s,
        poll_timeout_s=settings.outpaint_poll_timeout_s,
    )
    app.state.settings = settings
    app.state.cache = cache
    app.state.comfy = comfy
    app.state.outpaint = OutpaintService(
        cache,
        comfy,
        retry_after_s=settings.outpaint_retry_after_s,
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

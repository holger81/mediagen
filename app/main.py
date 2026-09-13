"""FastAPI entrypoint for mediagen."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from app.cache import MediaCache
from app.comfy_client import ComfyUiOutpaintClient, default_workflow_path
from app.config import Settings, get_settings
from app.outpaint import OutpaintService


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
    app.state.outpaint = OutpaintService(cache, comfy)
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


def _jpeg_response(result) -> Response:
    return Response(
        content=result.bytes,
        media_type="image/jpeg",
        headers={
            "X-Media-Hash": result.hash,
            "X-Cache": result.cache,
            "X-Outpaint-Source": result.source,
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
) -> Response:
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty image")
    service: OutpaintService = request.app.state.outpaint
    try:
        result = await service.outpaint(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return _jpeg_response(result)


@app.get("/v1/image/outpaint/{sha256}")
async def image_outpaint_lookup(sha256: str, request: Request) -> Response:
    if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256.lower()):
        raise HTTPException(status_code=400, detail="invalid sha256")
    service: OutpaintService = request.app.state.outpaint
    result = service.lookup(sha256.lower())
    if result is None:
        raise HTTPException(status_code=404, detail="not cached")
    return _jpeg_response(result)

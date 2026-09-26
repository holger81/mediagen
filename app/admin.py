"""Admin UI: cached outpaint gallery + recent logs."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from app.cache import MediaCache
from app.config import Settings
from app.log_buffer import get_log_handler
from app.outpaint import OutpaintService
from app.runtime_settings import load_runtime_settings, save_runtime_settings

router = APIRouter(tags=["admin"])

_ADMIN_HTML = Path(__file__).resolve().parent / "static" / "admin.html"


class AdminSettingsUpdate(BaseModel):
    flux_quality_gate: bool | None = Field(default=None)


def _check_admin(request: Request, token: str | None) -> None:
    settings: Settings = request.app.state.settings
    expected = (settings.admin_token or "").strip()
    if not expected:
        return
    provided = (token or "").strip()
    if not provided:
        provided = (request.headers.get("X-Admin-Token") or "").strip()
    if provided != expected:
        raise HTTPException(status_code=401, detail="admin auth required")


def _settings_payload(request: Request) -> dict:
    service: OutpaintService = request.app.state.outpaint
    settings: Settings = request.app.state.settings
    return {
        "flux_quality_gate": service.flux_quality_gate,
        "flux_quality_gate_env_default": settings.flux_quality_gate,
    }


@router.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request, token: str | None = Query(default=None)) -> HTMLResponse:
    _check_admin(request, token)
    if not _ADMIN_HTML.is_file():
        raise HTTPException(status_code=500, detail="admin page missing")
    return HTMLResponse(_ADMIN_HTML.read_text(encoding="utf-8"))


@router.get("/admin/api/settings")
async def admin_get_settings(
    request: Request,
    token: str | None = Query(default=None),
) -> JSONResponse:
    _check_admin(request, token)
    return JSONResponse(_settings_payload(request))


@router.put("/admin/api/settings")
async def admin_put_settings(
    request: Request,
    body: AdminSettingsUpdate,
    token: str | None = Query(default=None),
) -> JSONResponse:
    _check_admin(request, token)
    service: OutpaintService = request.app.state.outpaint
    cache: MediaCache = request.app.state.cache
    if body.flux_quality_gate is not None:
        service.set_flux_quality_gate(body.flux_quality_gate)
        stored = load_runtime_settings(cache.cache_dir)
        stored["flux_quality_gate"] = service.flux_quality_gate
        save_runtime_settings(cache.cache_dir, stored)
    return JSONResponse(_settings_payload(request))


@router.get("/admin/api/activity")
async def admin_activity(
    request: Request,
    token: str | None = Query(default=None),
) -> JSONResponse:
    _check_admin(request, token)
    service: OutpaintService = request.app.state.outpaint
    return JSONResponse(service.activity.snapshot())


@router.get("/admin/api/entries")
async def admin_entries(
    request: Request,
    token: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> JSONResponse:
    _check_admin(request, token)
    cache: MediaCache = request.app.state.cache
    service: OutpaintService = request.app.state.outpaint
    entries = cache.list_entries(limit=limit, offset=offset)
    inflight = service.inflight_hashes()
    inflight_set = set(inflight)
    return JSONResponse(
        {
            "total": cache.count(),
            "max_items": cache.max_items,
            "inflight": inflight,
            "entries": [
                {
                    "hash": e.hash,
                    "source": e.source,
                    "hits": e.hits,
                    "created_at": e.created_at,
                    "last_access": e.last_access,
                    "size_bytes": e.size_bytes,
                    "done": e.done,
                    "flux": e.flux,
                    "pads": e.pads,
                    "generating": e.hash in inflight_set,
                }
                for e in entries
            ],
        }
    )


@router.get("/admin/api/logs")
async def admin_logs(
    request: Request,
    token: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=500),
) -> JSONResponse:
    _check_admin(request, token)
    lines = get_log_handler().snapshot(limit=limit)
    return JSONResponse(
        {
            "lines": [
                {
                    "ts": line.ts,
                    "level": line.level,
                    "logger": line.logger,
                    "message": line.message,
                }
                for line in lines
            ]
        }
    )


@router.delete("/admin/api/entries/{sha256}")
async def admin_delete_entry(
    sha256: str,
    request: Request,
    token: str | None = Query(default=None),
) -> JSONResponse:
    _check_admin(request, token)
    digest = sha256.lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise HTTPException(status_code=400, detail="invalid sha256")
    cache: MediaCache = request.app.state.cache
    service: OutpaintService = request.app.state.outpaint
    service.cancel_inflight(digest)
    removed = cache.invalidate(digest)
    # Also clear a lone .done / .pads / .flux if the JPEG was already gone.
    if not removed:
        cache.done_marker_for(digest).unlink(missing_ok=True)
        cache.pads_marker_for(digest).unlink(missing_ok=True)
        (cache.cache_dir / f"{digest}.flux").unlink(missing_ok=True)
    return JSONResponse({"ok": True, "hash": digest, "removed": removed})


@router.get("/admin/cache/{sha256}.jpg")
async def admin_cache_jpeg(
    sha256: str,
    request: Request,
    token: str | None = Query(default=None),
) -> FileResponse:
    _check_admin(request, token)
    digest = sha256.lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise HTTPException(status_code=400, detail="invalid sha256")
    cache: MediaCache = request.app.state.cache
    hit = cache.get_by_hash(digest, touch=False)
    if hit is None:
        raise HTTPException(status_code=404, detail="not cached")
    return FileResponse(hit.path, media_type="image/jpeg", filename=f"{digest[:12]}.jpg")

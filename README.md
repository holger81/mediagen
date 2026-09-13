# Mediagen

Docker media-generation API. Starts with **image outpaint** (greatroom-wall Flux Fill flow). Layout is ready for later image generate/edit/inpaint and audio/video.

## Quick start (Portainer)

1. Deploy this repo as a stack with [`docker-compose.yml`](docker-compose.yml).
2. Ensure the existing ComfyUI stack is up (Flux fill models: `flux1-fill-dev`, DualCLIP, `ae.safetensors`) at `COMFYUI_BASE_URL`.
3. Create the host cache dir if needed: `/shared/mediagen/cache`.

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `CACHE_DIR` | `/cache` | Cache directory **inside** the container |
| `CACHE_HOST_DIR` | `/shared/mediagen/cache` | Host path bind-mounted to `CACHE_DIR` |
| `CACHE_MAX_ITEMS` | `1000` | Max cached outpaints (frequency-aware eviction) |
| `COMFYUI_BASE_URL` | `http://192.168.10.31:8188` | Existing ComfyUI HTTP API |
| `CORS_ORIGINS` | `*` | CORS allow list |

## API

- `POST /v1/image/outpaint` — multipart field `image`; returns JPEG
- `GET /v1/image/outpaint/{sha256}` — cache-only lookup
- `GET /health` — API + Comfy reachability

Response headers: `X-Media-Hash`, `X-Cache: hit|miss`, `X-Outpaint-Source: flux|local`.

### Example

```bash
curl -sS -X POST http://HOST:8090/v1/image/outpaint \
  -F image=@cover.jpg \
  -o outpaint.jpg -D -
```

## Cache behavior

- Key: `sha256(cache_version || source_bytes)` (`empty-prompt-feather0-v5`)
- Files under `CACHE_DIR` as `{hash}.jpg` + SQLite hit index
- Eviction: single-hit (probation) entries first; hot keys (`hits >= 2`) kept longer

## Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
CACHE_DIR=/tmp/mediagen-cache pytest
ruff check app tests
uvicorn app.main:app --reload --port 8090
```

## Outpaint pipeline

Matches ha_native_dash greatroom wall:

1. Content-hash cache lookup
2. Instant local edge pad (Pillow)
3. Skip Flux for uniform/black mattes
4. Else ComfyUI Flux Fill (`workflows/album_outpaint_api.json`)
5. Quality gate; reject invented mats / hard seams → keep local pad

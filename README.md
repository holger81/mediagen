# Mediagen

Docker media-generation API. Starts with **image outpaint** (greatroom-wall Flux Fill flow). Layout is ready for later image generate/edit/inpaint and audio/video.

## Quick start (Portainer)

1. Deploy this repo as a stack with [`docker-compose.yml`](docker-compose.yml).
2. Ensure the existing ComfyUI stack is up (Flux fill models: `flux1-fill-dev`, DualCLIP, `ae.safetensors`) at `COMFYUI_BASE_URL`.
3. Create the host cache dir if needed: `/shared/mediagen/cache`.

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `API_HOST_PORT` | `8090` | Host port published to the API |
| `API_HOST_BIND` | `0.0.0.0` | Host bind address for the published port |
| `CACHE_DIR` | `/cache` | Cache directory **inside** the container |
| `CACHE_HOST_DIR` | `/shared/mediagen/cache` | Host path bind-mounted to `CACHE_DIR` |
| `CACHE_MAX_ITEMS` | `1000` | Max cached outpaints (frequency-aware eviction) |
| `COMFYUI_BASE_URL` | `http://192.168.10.31:8188` | Existing ComfyUI HTTP API |
| `OUTPAINT_POLL_TIMEOUT_S` | `180` | Max seconds to wait for Comfy Flux (cold load can exceed 90s) |
| `OUTPAINT_RETRY_AFTER_S` | `5` | Suggested poll interval when status is `generating` |
| `CORS_ORIGINS` | `*` | CORS allow list |

### Portainer: “failed programming external connectivity”

That error on `mediagen-api-1` almost always means the **host port is already taken** (or a leftover container still holds it).

1. In Portainer → Stack → Environment, set `API_HOST_PORT` to a free port (e.g. `18090`).
2. Or on the Docker host: `docker rm -f mediagen-api-1` then redeploy.
3. Confirm nothing else owns the port: `ss -ltnp | grep 8090` (or your chosen port).

## API

### Outpaint (async)

- `POST /v1/image/outpaint` — multipart field `image`, optional layout fields
  - **Defaults** (omit layout): pads `256/128/256/128` (left/top/right/bottom)
  - **Pads form:** `pad_left`, `pad_top`, `pad_right`, `pad_bottom` (all four required)
  - **Canvas form:** `out_width`, `out_height`, `x`, `y` (place unscaled source at `x,y`; pads derived)
  - Do not mix forms. Translate only — source is never resized.
  - **Cache hit / fast local-only (uniform edges):** `200` JPEG
  - **Generation started (or already in flight):** `202 Accepted` JSON
- `GET /v1/image/outpaint/{sha256}` — poll by content hash (hash includes layout)
  - **Ready:** `200` JPEG
  - **Still generating:** `202` JSON
  - **Unknown:** `404`
- `GET /health` — API + Comfy reachability

**Ready (`200`) headers:** `X-Media-Hash`, `X-Cache: hit|miss`, `X-Outpaint-Source: flux|local`, `X-Outpaint-Status: ready`, `X-Outpaint-Pad: L,T,R,B`, `X-Outpaint-Size: WwHh`.

**Generating (`202`) body + headers:**

```json
{ "status": "generating", "hash": "<sha256>", "retry_after_s": 5 }
```

Headers: `Retry-After`, `X-Media-Hash`, `X-Outpaint-Status: generating`, `X-Cache: miss`, `X-Outpaint-Pad`, `X-Outpaint-Size`.

### Example

```bash
# First miss on a pictorial cover → 202
curl -sS -D - -o /tmp/out.json -X POST http://HOST:18090/v1/image/outpaint \
  -F image=@cover.jpg
HASH=$(python3 -c 'import json;print(json.load(open("/tmp/out.json"))["hash"])')

# Custom pads (or use out_width/out_height/x/y instead)
curl -sS -D - -o /tmp/out.jpg -X POST http://HOST:18090/v1/image/outpaint \
  -F image=@cover.jpg -F pad_left=64 -F pad_top=32 -F pad_right=64 -F pad_bottom=32

# Poll until ready
while true; do
  code=$(curl -sS -o /tmp/out.jpg -w '%{http_code}' \
    "http://HOST:18090/v1/image/outpaint/$HASH")
  [[ "$code" == "200" ]] && break
  sleep 5
done
file /tmp/out.jpg
```

## Cache behavior

- Key: `sha256(version || pads:L,T,R,B || decoded RGB fingerprint)` — same cover with
  different layout is a different cache entry
- After each generation attempt (Flux **or** local fallback), a `.done` marker is
  written — pictorial local pads are **not** wiped and re-queued on every play
  (this was regenerating Comfy jobs like `ha_album_outpaint_00367_` /
  `00420_` for the same cover)
- Files under `CACHE_DIR` as `{hash}.jpg` + `{hash}.pads` + SQLite hit index
- Eviction: single-hit (probation) entries first; hot keys (`hits >= 2`) kept longer
- Single-flight per hash: concurrent POSTs share one Comfy job and all get `202`

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

1. Content-hash cache lookup (includes layout pads)
2. Instant local edge pad (Pillow) — sync `200` for uniform/black mattes
3. Pictorial covers: background Flux Fill (`workflows/album_outpaint_api.json`); clients poll
4. Flux prompts steer against invented type (positive “no text…”, real negative CLIP encode)
5. Quality gate; reject invented mats / hard seams → keep local pad

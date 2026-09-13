FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libjpeg62-turbo zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY workflows ./workflows

ENV CACHE_DIR=/cache \
    CACHE_MAX_ITEMS=1000 \
    COMFYUI_BASE_URL=http://192.168.10.31:8188 \
    WORKFLOWS_DIR=/app/workflows \
    PYTHONUNBUFFERED=1

EXPOSE 8090

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090"]

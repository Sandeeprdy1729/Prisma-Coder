# PrismaCoder inference image: Python 3.11 + Node 20 (Node enables full V1-V5
# gates server-side). Build from the repo root:
#   docker build -f docker/Dockerfile -t prismacoder/inference:latest .
FROM node:20-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3-pip git curl ca-certificates \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install python deps first (layer cache)
COPY requirements.txt pyproject.toml ./
RUN pip install --break-system-packages -r requirements.txt \
    && pip install --break-system-packages \
        "fastapi>=0.115" "uvicorn[standard]>=0.30" "sse-starlette>=2.1" \
        "gradio>=4.44" "pydantic>=2.8" "huggingface-hub>=0.26"

# install the package + app code
COPY prismacoder ./prismacoder
COPY inference ./inference
COPY training ./training
COPY benchmark ./benchmark
RUN pip install --break-system-packages --no-deps .

# warm the npm cache for the verifier sandbox toolchain (best-effort, offline-safe)
RUN npm install --prefix /opt/pc-cache --no-audit --no-fund \
        typescript@5.6.3 prisma@7.10.0 @prisma/client@7.10.0 vitest@2.1.8 tsx@4.19.2 \
    || echo "npm warm-cache skipped (will install on first verify)"

ENV PRISMACODER_MODEL=prismacoder/PrismaCoder-3B \
    PRISMACODER_FULL_GATES=1 \
    PRISMACODER_SANDBOX_CACHE=/opt/pc-cache/node_modules

EXPOSE 8000 7860

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["api"]

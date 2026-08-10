FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    CONFIG_FILE=/app/conf/conf.yaml

WORKDIR /app

# Base dependencies
RUN apt-get update -o Acquire::Retries=5 \
 && apt-get install -y --no-install-recommends \
      ffmpeg git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Install uv via pip to avoid ghcr.io rate limiting on Railway
RUN pip install --no-cache-dir uv

# Install deps
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy source & install project
COPY . /app
RUN uv pip install --no-deps .

# Ensure startup script is executable
RUN chmod +x /app/scripts/start-app.sh

EXPOSE 12393

CMD ["/app/scripts/start-app.sh"]

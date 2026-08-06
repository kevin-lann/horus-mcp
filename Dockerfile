# linux/amd64 is pinned because the optional `libsql` (Turso) dependency only ships
# manylinux wheels for x86_64 -- an arm64 build would fall back to a from-source
# cargo/maturin build here and fail (no Rust toolchain in this image). Cloud Run
# itself only runs x86_64, so this also matches the deployment target.
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir '.[turso]'

# Kaleido (used to render charts to PNG) requires a real Chrome binary since
# kaleido 1.x -- it no longer bundles Chromium. Install Chrome's headless
# runtime deps plus a pinned Chrome-for-Testing build, and point kaleido at
# it via BROWSER_PATH so it's found regardless of which user/HOME runs it.
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    ca-certificates \
    fonts-liberation \
    libasound2t64 \
    libatk-bridge2.0-0t64 \
    libatk1.0-0t64 \
    libatspi2.0-0t64 \
    libcairo2 \
    libcups2t64 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libglib2.0-0t64 \
    libnspr4 \
    libnss3 \
    libpango-1.0-0 \
    libx11-6 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    libxshmfence1 \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/chrome-for-testing && plotly_get_chrome -y --path /opt/chrome-for-testing
ENV BROWSER_PATH=/opt/chrome-for-testing/chrome-linux64/chrome

EXPOSE 5050

# Create a non-root user and group for the app
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
RUN chown -R appuser:appgroup /app
RUN mkdir -p /data && chown -R appuser:appgroup /data
USER appuser

CMD ["horus-mcp-http"]

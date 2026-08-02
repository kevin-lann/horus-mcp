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

EXPOSE 5050

# Create a non-root user and group for the app
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
RUN chown -R appuser:appgroup /app
RUN mkdir -p /data && chown -R appuser:appgroup /data
USER appuser

CMD ["horus-mcp-http"]

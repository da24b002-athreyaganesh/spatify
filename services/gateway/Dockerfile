# syntax=docker/dockerfile:1
# Gateway: the FastAPI REST API. Build from the repo root:
#   docker build -f services/gateway/Dockerfile .
#
# Multi-stage: uv builds the virtual environment from uv.lock in the builder; the final
# image is python:slim plus that environment, with no uv, no build cache and no source.

FROM python:3.12.15-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first: this layer is reused until pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra serving --no-install-project
# Then the code, installed as a regular (non-editable) package.
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra serving --no-editable

FROM python:3.12.15-slim
RUN useradd --system --create-home --uid 10001 app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
# The gateway logs every request itself (JSON), so uvicorn's access log is off.
CMD ["uvicorn", "spatify.gateway.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]

# syntax=docker/dockerfile:1
# Frontend: the Streamlit web app (Discover, the operator console, Help). It talks only to
# the gateway, at API_BASE_URL. Build from the repo root:
#   docker build -f services/frontend/Dockerfile .
#
# Multi-stage, like the gateway: uv builds the environment from uv.lock in the builder, and
# the final image is python:slim plus that environment, running as a non-root user.

FROM python:3.12.15-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first: this layer is reused until pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra frontend --no-install-project
# Then the code, installed as a regular (non-editable) package.
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra frontend --no-editable

FROM python:3.12.15-slim
RUN useradd --system --create-home --uid 10001 app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1
USER app
# 8501: the web app; 9101: its Prometheus metrics.
EXPOSE 8501 9101
CMD ["python", "-m", "spatify.frontend"]

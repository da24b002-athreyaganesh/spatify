# syntax=docker/dockerfile:1
# Model server: serves the registry's champion with `mlflow models serve`, through a
# supervisor that restarts it when the alias moves (src/spatify/model_server.py).
# Build from the repo root:  docker build -f services/model_server/Dockerfile .
#
# The model brings its own code (code_paths) and needs only mlflow, numpy, pandas and
# pyarrow, all in the `mlflow-server` extra; models are served with --env-manager local.

FROM python:3.12.15-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra mlflow-server --no-install-project
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra mlflow-server --no-editable

FROM python:3.12.15-slim
RUN useradd --system --create-home --uid 10001 app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1
USER app
EXPOSE 8080
CMD ["python", "-m", "spatify.model_server"]

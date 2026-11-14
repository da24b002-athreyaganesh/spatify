# syntax=docker/dockerfile:1
# MLflow tracking server and model registry, backed by Postgres; artifacts in a volume,
# served through the server (--serve-artifacts). docker-compose.yml supplies the command.
# Build from the repo root:  docker build -f services/mlflow/Dockerfile .
#
# Same pinned mlflow as development and the model server (the `mlflow-server` extra).

FROM python:3.12.15-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra mlflow-server --no-install-project

FROM python:3.12.15-slim
RUN useradd --system --create-home --uid 10001 app \
    && mkdir -p /mlflow/artifacts && chown app:app /mlflow/artifacts
COPY --from=builder --chown=app:app /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1
USER app
EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]

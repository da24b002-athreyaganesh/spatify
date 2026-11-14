# syntax=docker/dockerfile:1
# Worker: watches data/incoming and the run queue, runs the DVC pipeline (Ray stages
# included) in the mounted repo, and records every run in Postgres.
# Build from the repo root:  docker build -f services/worker/Dockerfile .
#
# The image holds the dependencies only. The code comes from the mounted repo
# (PYTHONPATH=/repo/src), so the stages DVC runs are exactly the code DVC hashed as
# their dependencies, even after an edit without a rebuild (D36).

FROM python:3.12.15-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra pipeline --extra worker --no-install-project

FROM python:3.12.15-slim
# git: DVC and MLflow read the repo's history. uv: the python_env stage exports uv.lock.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv
COPY --from=builder /app/.venv /app/.venv
# The worker writes into the mounted repo, which belongs to the host user (uid 1000 on
# this drive), so it runs as a user with that uid. The user needs an account in the
# image, not just a number: DVC and MLflow look up the current user's name.
ARG UID=1000
ARG GID=1000
RUN groupadd --gid "$GID" worker \
    && useradd --uid "$UID" --gid "$GID" --create-home --shell /bin/sh worker
USER worker
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONPATH=/repo/src
WORKDIR /repo
EXPOSE 9102
CMD ["python", "-m", "spatify.worker"]

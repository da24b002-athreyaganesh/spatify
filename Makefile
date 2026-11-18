# Spatify's developer tasks (D43). `make help` lists them; run them from spatify/.
#   make ci          style, tests, then the pipeline and its promotion gate
#   make coverage    tests with a coverage report
#   make load-test   /recommend latency at 1, 10 and 40 clients   (the stack must be up)
#   make failure-test  kill a training run; the champion must keep serving   (stack up)

SHELL := /bin/bash
export RAY_ENABLE_UV_RUN_RUNTIME_ENV := 0   # Ray hangs under `uv run` without it (D15)
export MLFLOW_DISABLE_AGENT_HINT := 1

.PHONY: help ci lint test coverage pipeline dashboards api-docs load-test failure-test smoke

help:  ## list the targets
	@grep -hE '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | sed -E 's/:.*## /\t/' | expand -t 16

ci: lint test pipeline  ## everything CI checks: style, tests, then dvc repro and the gate

lint:  ## ruff: PEP 8 style, imports, naming and formatting
	uv run ruff check .
	uv run ruff format --check .

test:  ## the pytest suite (the promtool and amtool checks need Docker)
	uv run pytest

coverage:  ## the tests with line and branch coverage (terminal, and reports/coverage/)
	uv run pytest --cov --cov-report=term-missing:skip-covered --cov-report=html

pipeline:  ## dvc repro against the stack's MLflow, then the promotion gate's decision
	bash scripts/ci_pipeline.sh

dashboards:  ## rebuild the Grafana dashboards from monitoring/grafana/build_dashboards.py
	uv run python monitoring/grafana/build_dashboards.py

api-docs:  ## refresh docs/openapi.json and the LLD's API reference from the code
	uv run python scripts/export_api_docs.py

load-test:  ## /recommend latency at 1, 10 and 40 clients; writes reports/load_test/
	uv run python scripts/load_test.py

failure-test:  ## kill the train stage mid-run; the champion must keep serving
	uv run python scripts/failure_test.py

smoke:  ## end-to-end check of the running stack through the gateway
	uv run python scripts/smoke_test.py

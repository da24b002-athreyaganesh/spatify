#!/usr/bin/env bash
# `make pipeline`, the last step of `make ci` (D43): reproduce the pipeline with DVC, then
# report the promotion gate's decision (D22). Not promoting is a normal outcome.
#
# It logs to the stack's MLflow (D25), so the stack must be up. The worker is stopped
# for the run and started again afterwards: two DVC processes can't share a repo (D36),
# and a stop waits for a pipeline run in progress to finish.
set -euo pipefail
cd "$(dirname "$0")/.."

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"
export SPATIFY_TRIGGER="${SPATIFY_TRIGGER:-ci}"
if ! curl -sf "$MLFLOW_TRACKING_URI/health" >/dev/null; then
  echo "MLflow isn't answering at $MLFLOW_TRACKING_URI: start the stack (docker compose up -d)."
  exit 1
fi

restart_worker=""
if docker compose ps --status running --services 2>/dev/null | grep -qx worker; then
  echo "Stopping the worker for the pipeline run (it finishes any run in progress first)."
  docker compose stop worker >/dev/null
  restart_worker=1
fi
trap 'if [ -n "$restart_worker" ]; then docker compose start worker >/dev/null; echo "Worker started again."; fi' EXIT

uv run dvc repro
echo
echo "== Changes against the last commit (dvc metrics diff)"
uv run dvc metrics diff --md || true
echo
echo "== The promotion gate (metrics/register.json)"
uv run python - <<'PY'
import json
r = json.load(open("metrics/register.json"))
metric = r.get("metric", "genre_match_at_k")
if r["promoted"]:
    print(f"Version {r['version']} ({r['variant']}) promoted on {metric}: "
          f"{r['challenger_score']} against {r['champion_score']} (min gain {r['min_gain']}).")
else:
    print(f"Version {r['version']} ({r['variant']}) not promoted on {metric}: gain {r['gain']} "
          f"< {r['min_gain']}; version {r['champion_version']} keeps serving.")
PY

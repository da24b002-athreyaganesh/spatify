# Spatify test plan

How Spatify is tested, what each test checks, and the acceptance criteria the system must meet. Results are in [test_report.md](test_report.md); the decisions behind the approach are D43 (testing and CI) and D44 (security) in the [decision log](decision_log.md).

## 1. Scope and approach

Spatify is tested at seven levels. The lower levels run on every change with one command (`make test`); the upper ones need the running stack and produce the evidence files in `evidence/`.

| Level | What it covers | Runs with | When |
|---|---|---|---|
| Unit and component | Each module against fakes: validation rules, PSI/KS, the recommender, feedback learning, the gateway's API, the worker's decisions, the frontend's pages, the monitoring config, the documents | `make test` (pytest, 172 tests) | Every change; part of `make ci` |
| Integration | Stages working together on real data formats: the Ray data stages; the whole pipeline into a real (throwaway) MLflow registry | `make test` | Every change |
| System | The running Compose stack, through its public interfaces only | `make smoke`, the runbook's batch drills | After a build |
| Performance | `/recommend` latency under 1, 10 and 40 concurrent clients | `make load-test` | Before submission |
| Reliability | A pipeline stage killed mid-run; services down | `make failure-test`; stopping a service | Before submission |
| Usability | First-time users finding music without help | In person (section 5) | Before submission |
| Code quality and security | Style, coverage, keys and logs | `make lint`, `make coverage`, tests | Every change |

`make ci` runs, in order: ruff (style), pytest, then `dvc repro` against the stack's MLflow and the promotion gate's decision. Any failure stops it.

**Simulated listeners.** Most feedback comes from the user simulator (D38); tests that need likes and skips use it, and the report says so.

## 2. Environment

- One laptop: 20 CPUs, 15 GB RAM (about 6 GB used by a browser and an editor), Ubuntu, Docker 29 with Compose.
- Python 3.12 from `uv.lock`; the same lock builds the Docker images and `python_env.yaml`.
- Data: the Spotify Tracks Dataset (`docs/data.md`) and the batches generated from it (`data/batches/`).
- Tests that run promtool or amtool need Docker and the pinned images; without them those 4 tests are skipped, not failed.

## 3. Acceptance criteria

The system is accepted when every criterion is met. Each comes from the problem statement (PS: FR = functional requirement, NFR = non-functional) or the evaluation rubric.

| ID | Criterion | Source | Measured by |
|---|---|---|---|
| AC-01 | A seed song, sliders, or both return K ranked recommendations; an unknown song or an empty query gets a clear error, not a crash. | FR1 | TC-05, TC-10, TC-13, TC-20 |
| AC-02 | Every like and skip is stored with its time and the model version that made the list; feedback on a list the gateway didn't serve is refused. | FR2 | TC-10, TC-20 |
| AC-03 | New batches are validated; a batch with more than 5% bad rows is quarantined and never reaches training. | FR3 | TC-02, TC-12, TC-17, TC-24 |
| AC-04 | Drift is computed per feature (PSI, KS) and exposed as metrics: the shifted batch is flagged, normal batches aren't. | FR4 | TC-04, TC-22, TC-23 |
| AC-05 | Drift or 2,000 new ratings trigger a retrain, and a candidate is promoted only if it beats the champion by the minimum gain. | FR5 | TC-09, TC-12, TC-18, TC-25 |
| AC-06 | The console shows run history, success and failure counts, the active model, drift status and alerts, and can roll back. | FR6, rubric (pipeline console) | TC-13, TC-20 |
| AC-07 | `/recommend` p95 latency ≤ 300 ms with no errors at 1 and 10 concurrent clients; at 40, no errors (latency reported). | NFR performance | TC-29 |
| AC-08 | A pipeline run that fails mid-way doesn't disrupt serving: no failed recommendations, same champion. | NFR reliability | TC-12, TC-24, TC-30 |
| AC-09 | A fresh clone, `dvc pull` and `dvc repro` reproduce the model metrics (only timings differ). | NFR reproducibility, O4 | TC-18, TC-19 |
| AC-10 | Every component is scraped by Prometheus (all targets UP), the dashboards show live data, and alerts reach the console when they fire and when they resolve. | NFR observability, O7 | TC-14, TC-15, TC-27, TC-28 |
| AC-11 | `docker compose up -d --build` starts the whole system (11 services) with one command. | NFR portability | TC-21 |
| AC-12 | First-time users reach their first recommendations in about 30 s without help (median ≤ 30 s, none over 60 s). | NFR usability, O9 | TC-32 |
| AC-13 | Ruff reports nothing, every automated test passes, and coverage (lines and branches) is at least 80%. | NFR maintainability, rubric | TC-34, TC-35 |
| AC-14 | Admin actions refuse a missing or wrong key, and no key appears in logs or the repository. | Guidelines (security) | TC-16, TC-36, TC-37 |
| AC-15 | After a forced bad deployment, rolling back puts the previous model back in service within about 15 s. | PS evaluation plan | TC-11, TC-26 |

## 4. Test cases

**Automated** cases run with `make test`; the test file is given. **Stack** cases need the running stack; their evidence file is given. Expected results are what passing looks like.

### Unit and component (automated)

| ID | What it checks | Where | Expected result | AC |
|---|---|---|---|---|
| TC-01 | Batch generator: normal, shifted (held-out genres) and drifted batches; planted bad rows; seeded output | `tests/test_batches.py` | Batches as specified; same seed, same batches | 03, 04 |
| TC-02 | Validation: schema, types, ranges, nulls, duplicates, a reason per rejected row, the 5% limit per source | `tests/test_validate.py` | Bad rows rejected with reasons; a source over 5% fails | 03 |
| TC-03 | Feature engineering: keys on the circle of fifths (related keys close, minor keys on their relative major), a track in several genres merged into one row, integer columns kept | `tests/test_features.py` | Expected columns and values | 01 |
| TC-04 | PSI and KS on distributions with a known answer; the drift rule (any major, or two moderate) | `tests/test_drift.py` | Identical data: PSI ≈ 0; shifted data: the expected level | 04 |
| TC-05 | Recommender: cosine neighbours, the seed and repeats of a song left out, sliders, Fisher weights, the held-out split | `tests/test_recommender.py` | K results in order; unknown seeds and empty queries rejected | 01 |
| TC-06 | "Why" labels: the traits a seed and a track share, ordered by their share of the score and capped; tiny shares left out; slider queries explain only the sliders set | `tests/test_explain.py` | As stated; an unknown seed is an error, as in /recommend | 01 |
| TC-07 | Feedback learning: weights recover the simulated taste; held-out NDCG beats cold start; too little or one-sided feedback gives no model | `tests/test_feedback.py` | As stated | 05 |
| TC-08 | Simulated listeners: like probability falls with distance; seeded queries | `tests/test_simulation.py` | As stated | 05 |
| TC-09 | Packaged model answers through MLflow pyfunc; the promotion rule | `tests/test_registry.py` | Same answers as the recommender; promote only with gain ≥ min gain | 05 |
| TC-10 | Gateway API against fake services: every endpoint, 422 on bad input, friendly 503s, feedback tied to its list, exploration slots, drift and alert endpoints, `/health` | `tests/test_gateway.py` | Status codes and bodies as in the LLD | 01, 02, 06 |
| TC-11 | Model-server supervisor: serves the alias, switches when it moves, restarts a crashed server, keeps serving when MLflow is down; its metrics | `tests/test_model_server.py` | As stated | 15 |
| TC-12 | Worker decisions: new data, drift, the feedback trigger, manual runs, quarantine, failed runs (no retry storms), re-added batches, throughput gauges | `tests/test_worker.py` | As stated; a failure never moves the champion | 03, 05, 08 |
| TC-13 | Frontend pages, headless: Discover (search, sliders, like/skip, wildcard), the console (pipeline, models, rollback with key, drift, monitoring), Help; a friendly message when the gateway is down | `tests/test_frontend.py` | No exceptions; the right calls and messages | 01, 06 |
| TC-14 | Monitoring config: every target a service, memory limits, localhost-only ports, every PromQL query attributed and reading a real metric, dashboards up to date and following the 7 commandments, promtool and amtool accept the config | `tests/test_monitoring.py` | As stated | 10 |
| TC-15 | Alert rules fire at the right time with the right text (promtool unit tests, run by TC-14) | `monitoring/prometheus/tests/rules_test.yml` | 10 scenarios pass | 10 |
| TC-16 | The admin key never reaches the logs (right or wrong key, header or Bearer) | `tests/test_gateway.py::test_the_admin_key_never_reaches_the_logs` | No log line contains the key | 14 |

### Integration (automated)

| ID | What it checks | Where | Expected result | AC |
|---|---|---|---|---|
| TC-17 | The three Ray stages end to end on a generated catalogue, with planted bad rows; a batch over the 5% limit stops the pipeline; identical reruns give identical files | `tests/test_pipeline.py` | Bad rows rejected; one row per track; the bad batch fails validate; same files on rerun | 03, 09 |
| TC-18 | The whole pipeline on a small sample, as `dvc repro` runs it: ingest → validate → features, drift_check → train → evaluate → register, into a throwaway MLflow registry; the champion answers a query; an equally good model isn't promoted | `tests/test_full_pipeline.py` | Version 1 promoted and serving; version 2 not promoted (gain 0) | 05, 09 |
| TC-19 | Reproducibility drill: fresh clone, `dvc pull`, `dvc repro --force` | Runbook step 0.5; `evidence/phase2_reproducibility_drill*.{log,txt}` | Metrics identical except timings | 09 |

### System (the running stack)

| ID | What it checks | How | Evidence | AC |
|---|---|---|---|---|
| TC-20 | Smoke test through the gateway: health, search, recommend, feedback, operator endpoints, error paths | `make smoke` | `evidence/phase3_smoke_test.log` | 01, 02, 06 |
| TC-21 | One command starts everything | `docker compose up -d --build`, then `docker compose ps` | `evidence/phase3_stack.txt`, `screenshots/phase6_prometheus_targets.png` | 11 |
| TC-22 | A normal batch: checked, no drift, no retrain | Runbook 5.2 | `evidence/phase5_worker.log` | 04 |
| TC-23 | A shifted batch: drift flagged, retrain, `DataDrift` alert | Runbooks 5.2 and 6.3 | `evidence/phase5_worker.log`, `screenshots/phase6_alert_firing.png` | 04, 05, 10 |
| TC-24 | A corrupt batch: quarantined, run failed, champion still serving | Runbook 5.2 | `screenshots/phase5_failed_run_still_serving.png` | 03, 08 |
| TC-25 | 2,000 new ratings: feedback retrain, the gate decides, a promotion is served | Runbooks 5.4 and 6.3 | `evidence/phase5_worker_feedback.log`, `evidence/phase5_retrain_to_redeploy.log` | 05 |
| TC-26 | Forced bad model, like rate drops, rollback from the console | Runbook 5.4 | `evidence/phase5_force_bad.log`, `screenshots/phase5_like_rate_drop.png`, `screenshots/phase5_rollback.png` | 15 |
| TC-27 | All Prometheus targets up; dashboards live under load | Runbook 6.3 | `screenshots/phase6_prometheus_targets.png`, `screenshots/phase6_dashboard_*.png` | 10 |
| TC-28 | Alerts reach the console, firing then resolved | Runbook 6.3 | `evidence/phase6_alerts.json`, `screenshots/phase6_console_alerts.png` | 06, 10 |

### Performance, reliability, usability

| ID | What it checks | How | Evidence | AC |
|---|---|---|---|---|
| TC-29 | Load test: `/recommend` at 1, 10 and 40 closed-loop clients, 30 s each, after a 5 s warm-up | `make load-test` | `evidence/phase7_load_test.log`, `reports/load_test/`, `screenshots/phase7_load_test_plot.png` | 07 |
| TC-30 | Failure test: the train stage killed (SIGKILL) mid-run while a listener asks for recommendations 5 times a second | `make failure-test` | `evidence/phase7_failure_test.log`, `screenshots/phase7_failure_test.png` | 08 |
| TC-31 | Gateway down: the app shows a friendly message, not a stack trace | Stop the gateway (runbook 4.2) and TC-13 | `screenshots/phase4_gateway_down.png` | 01 |
| TC-32 | First-time users: time from seeing Discover to the first list, without help | Section 5 | `evidence/phase7_usability_session.md` | 12 |
| TC-33 | The layout works at phone width | Browser device mode | `screenshots/phase4_mobile.png` | 12 |

### Code quality and security

| ID | What it checks | How | Expected result | AC |
|---|---|---|---|---|
| TC-34 | Style (PEP 8 rules, imports, naming) and formatting | `make lint` | No findings | 13 |
| TC-35 | Coverage of `src/spatify` by the automated tests | `make coverage` (`evidence/phase7_coverage.txt`) | ≥ 80% (lines and branches) | 13 |
| TC-36 | Rollback, trigger and alert endpoints refuse a missing or wrong key (`401`) | `tests/test_gateway.py`, `tests/test_frontend.py` | `401`, and the console explains it | 14 |
| TC-37 | No secrets in the repository | `git ls-files \| grep -x .env` prints nothing; `.env` is `0600` | As stated | 14 |
| TC-38 | The documents agree with the code: the LLD's API reference and `openapi.json` match the gateway; every endpoint is documented; every link resolves; every cited decision exists | `tests/test_docs.py` | As stated | 13 |

## 5. Usability test procedure (TC-32)

1. Two or three people who haven't seen Spatify; no names recorded (testers A, B, C).
2. Open Discover in a clean browser window and say only: "Find some music you'd like."
3. Time from first seeing the page to the first recommendation list; note where they hesitated and anything that confused them. Don't help.
4. Record each tester's time and notes in `evidence/phase7_usability_session.md`.

## 6. Entry and exit criteria

- **Entry:** the code builds (`docker compose build`), `make lint` and `make test` pass, and the stack is up with a champion serving.
- **Exit:** every acceptance criterion is met, or a failure is recorded in the report with its cause and a fix. The report quotes the totals (tests run, passed, failed, skipped) and the evidence for each criterion.

## 7. Known limits of the testing

- **Simulated feedback.** Listeners are simulated (D38), so feedback-learning results show that the loop works, not how much real listeners would gain.
- **The Postgres stores are tested through fakes.** Their SQL was checked against the running database inside rolled-back transactions, and the stack tests (TC-20 to TC-30) run it for real.
- **One machine.** The load test measures this laptop; the gateway and model server each run one worker process.

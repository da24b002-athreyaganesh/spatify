# Spatify test report

Results of the tests in the [test plan](test_plan.md), from the runs on 2026-11-18 (REMAINING_PHASES steps 7.2 and 7.3) and the evidence of the earlier phases. Every number below is quoted from the evidence file named next to it.

- **Date:** 2026-11-18 (stack, performance and reliability runs between about 18:20 and 18:35 IST; the failure-test rerun and the usability test later that evening)
- **Commit:** `eecfa0a` plus the uncommitted Phase 7 changes (the test files, scripts and documents listed in AI_DISCLOSURE)
- **Machine:** the laptop in the test plan (20 CPUs, 15 GB RAM), Docker 29, Python 3.12
- **Champion during the stack tests:** version 8 (`all-feedback`, promoted by the feedback retrain of run 244)

## 1. Summary

| | Total | Passed | Failed | Not run |
|---|---:|---:|---:|---:|
| Automated tests (`make test`) | 166 | 166 | 0 | 0 |
| Stack, performance, reliability and usability cases (TC-19 to TC-33) | 15 | 14 | 1 (TC-32, usability) | 0 |
| Acceptance criteria | 15 | 14 met | 1 not met (AC-12) | |

**Verdict: accepted except for usability.**
- **AC-12 is not met:** three first-time users took 2 min 13 s to 3 min 20 s to reach their first recommendations, against a target of about 30 seconds (section 5).
- **Every other criterion is met.** AC-10 was met after a fix: the failure-run alert now fires (section 4).

Testing found four new defects, all fixed (section 7).

## 2. Automated tests

From `evidence/phase7_coverage.txt` (`make coverage`) and `evidence/phase7_make_ci.log` (`make ci`): **166 passed, 0 failed, 0 skipped** in both runs. Counts are pytest's: a parametrized test counts once per case. The promtool and amtool checks ran, since Docker was available.

| Suite | Test file | Tests | Passed | Failed | Skipped |
|---|---|---:|---:|---:|---:|
| Batch generator | `test_batches.py` | 6 | 6 | 0 | 0 |
| Validation | `test_validate.py` | 11 | 11 | 0 | 0 |
| Features | `test_features.py` | 5 | 5 | 0 | 0 |
| Drift (PSI, KS) | `test_drift.py` | 8 | 8 | 0 | 0 |
| Recommender | `test_recommender.py` | 13 | 13 | 0 | 0 |
| "Why" labels | `test_explain.py` | 6 | 6 | 0 | 0 |
| Feedback learning | `test_feedback.py` | 8 | 8 | 0 | 0 |
| Simulated listeners | `test_simulation.py` | 7 | 7 | 0 | 0 |
| Packaged model and gate | `test_registry.py` | 5 | 5 | 0 | 0 |
| Gateway API | `test_gateway.py` | 34 | 34 | 0 | 0 |
| Model-server supervisor | `test_model_server.py` | 10 | 10 | 0 | 0 |
| Pipeline worker | `test_worker.py` | 16 | 16 | 0 | 0 |
| Frontend | `test_frontend.py` | 19 | 19 | 0 | 0 |
| Monitoring config (incl. promtool, amtool) | `test_monitoring.py` | 14 | 14 | 0 | 0 |
| Ray data stages (integration) | `test_pipeline.py` | 3 | 3 | 0 | 0 |
| Whole pipeline (integration) | `test_full_pipeline.py` | 1 | 1 | 0 | 0 |
| **Total** | | **166** | **166** | **0** | **0** |

The alert rules' own unit tests (10 scenarios) run inside `test_monitoring.py`. After these runs, six tests were added: one for defect 6 (`test_worker.py`), four that check the documents against the code (`test_docs.py`, TC-38), and one for defects 8 and 9 (`test_frontend.py`). The suite is now 172 tests, all passing.

**Coverage: 84%** of `src/spatify`, lines and branches (2,750 statements, 486 branches), against a target of ≥ 80%. The modules below 80%:

| Module | Coverage | Why |
|---|---:|---|
| `worker/store.py` | 0% | Postgres queries; the worker's tests use an in-memory fake. Its SQL was checked against the real database in rolled-back transactions and runs in every stack test. |
| `gateway/store.py` | 38% | The same, for the gateway's queries. |
| `worker/__main__.py`, `frontend/app.py` | 0% | Start-up code that runs only inside the containers; the pages and the worker loop are tested directly. |
| `worker/runner.py` | 65% | The lines that start a real `dvc` process; the worker's tests use a fake DVC. |
| `model/tracking.py` | 67% | The git and `dvc.lock` lookups, which return "unknown" outside a repo. |
| `model/plots.py` | 69% | The feedback chart with data (the tests' samples have no feedback). |
| `batches.py` | 73% | The command-line entry point; the generator itself is tested. |
| `model_server.py` | 76% | `main()` and the real `mlflow models serve` launch; the supervisor's logic is tested with fakes. |

**Style:** `make lint` reported nothing: "All checks passed!", and 87 files already formatted (`evidence/phase7_make_ci.log`).

**`make ci`'s pipeline step** (`evidence/phase7_make_ci.log`):
- The data stages were up to date.
- train, evaluate and register reran, because the failure test's killed run had left a newer feedback snapshot.
- Register made version 11 (`all-feedback`), **not promoted on feedback_ndcg: gain 0.0015 < 0.01; version 8 keeps serving.** That is the gate working as designed.

## 3. Performance (TC-29)

`make load-test`: 30 s per level after a 5 s warm-up, in a closed loop, with realistic listener queries (`evidence/phase7_load_test.log`, `reports/load_test/load_test.md`):

| clients | requests | req/s | p50 ms | p95 ms | p99 ms | errors | Little's R = N/X (ms) | p95 ≤ 300 ms, no errors |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 1,461 | 48.7 | 19.4 | **33.3** | 44.5 | 0 | 20.5 | yes |
| 10 | 2,941 | 98.0 | 99.9 | **159.6** | 193.0 | 0 | 102.0 | yes |
| 40 | 3,320 | 110.7 | 356.8 | **516.0** | 585.7 | 0 | 361.4 | no (not required) |

Plot: `screenshots/phase7_load_test_plot.png`. Grafana during the 40-client level: `screenshots/phase7_grafana_during_load.png`.

**Reading:**
- **The service time is about 20 ms:** the time for one client's request, covering the gateway, the model server, the "why" labels and saving the list in Postgres.
- **Throughput levels off at about 100–110 requests a second:**
  - From 1 to 10 clients it doubles (48.7 to 98.0).
  - From 10 to 40 clients it rises only 13% (to 110.7), while the median latency rises 3.6 times.
- **The extra latency is queueing, not slower work.** Little's law (R = N/X) predicts 20.5, 102.0 and 361.4 ms at the three levels, and the measured medians are 19.4, 99.9 and 356.8 ms. The limit is that the gateway and the model server each run one process.
- **Every request was answered at every level** (7,722 in all, 0 errors), all by version 8.
- **No alert:** in Grafana, p95 crossed the 300 ms line during the 40-client level. `RecommendLatencyHigh` didn't fire, because that level lasted 30 s and the alert needs 2 minutes.

**AC-07 is met:** p95 was 33.3 ms at 1 client and 159.6 ms at 10, with no errors at any level. At 40 clients the p95 of 516 ms is reported, as the criterion asks; it doesn't need to be under 300 ms.

If 40 clients had to be under 300 ms, the next step would be:
1. More processes: two gateway workers (`uvicorn --workers 2`, about 120 MB more), and `WORKERS=2` on the model server. Each model-server worker holds its own copy of the model (about 280 MB), so its memory limit would go up to about 1 GB.
2. Then measure again.

## 4. Reliability (TC-30)

`make failure-test` (`evidence/phase7_failure_test.log`): a manual retrain was queued (run 248), and the train stage was killed 20.6 s later. Meanwhile a listener asked for recommendations five times a second.

| Check | Result |
|---|---|
| The run is recorded as failed | **PASS**: "Retraining failed at train; the champion keeps serving." |
| Its train stage is the one marked failed | **PASS** |
| The champion is the same version | **PASS**: version 8 before and after |
| Every recommendation was answered, by the champion | **PASS**: 110 requests, 0 failed, all by version 8 |

**Result: PASS, 4 of 4.** An earlier attempt (run 247, visible in `screenshots/phase7_failure_test.png`) also failed at train and also left version 8 serving; the log is of the second run.

**Rerun after the fix for defect 6** (`evidence/phase7_failure_test_rerun.log`):
- The worker was restarted, and the test ran twice more (runs 249 and 250). The train stage was killed each time; the log is of run 250.
- Run 250 passed all 4 checks again: 49 requests, 0 failed, all answered by version 8.
- **This time `PipelineRunFailed` fired** at 13:45:45 UTC, after run 249, and resolved by itself at 14:03:55. The console showed "A pipeline run failed" (`screenshots/phase7_failure_alert.png`).

**AC-08 is met.** The other kind of failure points the same way: in run 241, validation rejected a corrupt batch (21.6% bad rows), the batch was quarantined, and the champion kept serving (`screenshots/phase5_failed_run_still_serving.png`).

## 5. Usability (TC-32)

Three people who hadn't seen Spatify were asked only to "find some music you'd like" (`evidence/phase7_usability_session.md`):

| Tester | Time to the first list | Where they hesitated, or what confused them |
|---|---:|---|
| A | 2 min 56 s (176 s) | not recorded |
| B | 2 min 13 s (133 s) | not recorded |
| C | 3 min 20 s (200 s) | not recorded |

**Median 176 s; all three over 60 s. AC-12 is not met:** first-time users took about six times the target of 30 seconds.

**Why:** the evidence doesn't record where the testers hesitated, so the cause isn't known. The page suggests three likely causes, to confirm with notes from the next session:
1. **The button is out of sight.** On a laptop screen, **Get recommendations** sits below all six sliders and "How many songs". After picking a song, a tester has to scroll to find it; Figure 1 of the user manual shows the first screen without it.
2. **Picking a song takes two steps:** type, then choose from a second list.
3. **Six sliders, each with a switch, are a lot to read** before doing anything.

**Suggested fix** (`src/spatify/frontend/views/discover.py`):
- Show recommendations as soon as a song is picked, or put the button directly under the song picker.
- Fold the sliders under "Or describe the feel instead", so the first screen is just the search box.
- Then run the test again with three new people, noting where each one hesitates.

## 6. Acceptance criteria

| ID | Criterion (short) | Result | Evidence |
|---|---|---|---|
| AC-01 | Recommendations from a seed, sliders or both; clear errors | **Met** | TC-05, TC-10, TC-13 pass; `evidence/phase3_smoke_test.log`; 7,722 load-test requests, 0 errors |
| AC-02 | Feedback stored with time and model version | **Met** | TC-10 passes; `evidence/phase3_smoke_test.log` |
| AC-03 | Bad batches quarantined, never trained on | **Met** | TC-02, TC-12, TC-17 pass; run 241 (21.6% bad rows, quarantined); `screenshots/phase5_failed_run_still_serving.png` |
| AC-04 | Drift per feature, as metrics; shifted flagged, normal not | **Met** | TC-04 passes; run 239 (normal_01: no drift), runs 240 and 245 (shifted_01: drift); `screenshots/phase6_dashboard_data.png` |
| AC-05 | Retrain on drift or 2,000 ratings; promote only if better | **Met** | TC-09, TC-18 pass. Run 243: version 6 not promoted (gain 0.0); run 244: version 8 promoted (gain 0.085); `make ci`: version 11 not promoted (gain 0.0015 < 0.01). `evidence/phase5_worker_feedback.log`, `evidence/phase7_make_ci.log` |
| AC-06 | Console: runs, counts, active model, drift, alerts, rollback | **Met** | TC-13 passes. `screenshots/phase7_failure_test.png` shows the run history with its counts (7 succeeded, 3 failed). `screenshots/phase4_console_models.png`, `screenshots/phase6_console_alerts.png` |
| AC-07 | p95 ≤ 300 ms, no errors at 1 and 10 clients; no errors at 40 | **Met** | Section 3: p95 33.3 ms and 159.6 ms; 0 errors at 1, 10 and 40 |
| AC-08 | A failed run doesn't disrupt serving | **Met** | Section 4: PASS 4 of 4 (110 requests, 0 failed); run 241 |
| AC-09 | A fresh clone reproduces the metrics | **Met** | `evidence/phase2_reproducibility_drill_metrics.txt`: only timings differ, plus `register.json`, because the fresh registry promotes version 1 |
| AC-10 | All targets up, live dashboards, alerts delivered | **Met** (after a fix) | All 10 targets UP (`screenshots/phase6_prometheus_targets.png`), and the dashboards are live (`screenshots/phase6_dashboard_*.png`). `DataDrift` fired at 11:03:40 UTC and resolved (`evidence/phase6_alerts.json`). `PipelineRunFailed` didn't fire at first (defect 6); after the fix it fired at 13:45:45 UTC and resolved at 14:03:55 (`screenshots/phase7_failure_alert.png`) |
| AC-11 | One command starts all 11 services | **Met** | `evidence/phase6_memory.txt` lists all 11 containers running after `docker compose up -d --build` |
| AC-12 | First-time users: about 30 s to the first list | **Not met** | Section 5: 133–200 s, median 176 s (`evidence/phase7_usability_session.md`) |
| AC-13 | Ruff clean, all tests pass, coverage ≥ 80% | **Met** | Section 2: no findings, 166 of 166, 84% |
| AC-14 | Admin actions need the key; no key in logs or the repo | **Met** | TC-16 and TC-36 pass; `git ls-files` doesn't list `.env`, and `.env` is `0600` (TC-37) |
| AC-15 | Rollback after a bad deployment within about 15 s | **Met** | `evidence/phase5_force_bad.log`, `screenshots/phase5_rollback.png`. The model server re-reads the alias every 10 s, and answered 4.8 s after starting a version (measured at version 8's promotion). |

## 7. Defects found during testing

Every defect testing found, and what became of it. Each fixed one has a test that would catch it again, except where noted.

| # | Found by | Defect | Fix | Test |
|---|---|---|---|---|
| 1 | Phase 5A first run | The worker image had no user account for uid 1000; the worker failed and retried about once a second (236 runs). | Account created in the image; failed runs no longer retried at full speed (D36) | `test_worker.py` |
| 2 | Phase 5B, checking the simulator against the gateway | The simulator treated the gateway's `201 Created` for feedback as an error. | Accept any 2xx | None automated: checked by running the simulator against the gateway app in-process |
| 3 | Step 6.3 (a Prometheus target down) | The frontend's metrics server started only when someone opened the app, so `ServiceDown` fired after every restart. | Start it with the process (D40) | `test_frontend.py::test_metrics_are_served_from_start_up_not_from_the_first_page_view` |
| 4 | Step 6.3 (a batch ignored) | A batch dropped in again after being removed was skipped, because its name and checksum matched the removed record. | Treat a returning batch as new (D36) | `test_worker.py::test_a_removed_batch_dropped_in_again_is_checked_again` |
| 5 | After a reboot | The drive came back at a new path; the containers and the venv kept the old one, and the worker crash-looped. | Recreate the containers and reinstall the venv after a move (D1) | Documented in the runbook ("Committing") |
| 6 | Step 7.2 (failure test) | `PipelineRunFailed` never fired. A labelled counter series first appears at 1, not 0, so `increase()` saw no change; after the worker restarted between the two attempts, each failed run started a new series at 1. | The worker exports every trigger and result at 0 when it starts (D41) | `test_worker.py::test_failed_runs_are_counted_from_zero_so_the_first_one_alerts`; re-run `make failure-test` to see the alert fire |
| 7 | Step 7.2 (`screenshots/phase7_grafana_during_load.png`) | The "share of requests failing (5xx)" panel's axis read 0% to 10000% when there were no errors, because Grafana's default range for a share is 0 to 100. | The axis tops out at 10% unless errors go higher (D42) | `test_monitoring.py` (the dashboards are rebuilt from the fixed builder); Grafana picked it up without a restart |
| 8 | Step 8.3 (the user manual's screenshots) | The console's pipeline diagram showed "ingest\n39,223 rows/s" with a literal `\n`: the line break was escaped twice. | Escape the text, then join the lines with DOT's line break | `test_frontend.py::test_dag_diagram_colours_stages_by_result` |
| 9 | Step 8.3 (the user manual's screenshots) | The console's charts squeezed their bars into a few pixels and clipped their titles; the like-rate chart lost a version's label. Streamlit gives a chart without its own sizing the type "fit", which fits the title and axes into the declared height. | Each chart fits the width only (`fit-x`); its height is the plotting area | `test_frontend.py::test_charts_fit_the_width_and_keep_their_height` |

**A finding, not a defect:** the promotion gate's held-out NDCG favours the champion, because the held-out lists are the champion's own picks (D37). The first feedback model wasn't promoted; the second (version 8, with more ratings) was; later ones (versions 9–11) weren't. A live A/B comparison of like rates would remove the bias (Stretch).

## 8. Conclusion

**14 of the 15 acceptance criteria are met; one is not.**
- **Met:** recommendation, feedback, validation, drift, retraining with a gate, the console, performance at 1 and 10 clients, reliability, reproducibility, observability (after the fix for defect 6), portability, code quality, security and rollback.
- **AC-12 (usability), not met:** first-time users took about three minutes, not 30 seconds, to their first recommendations. Section 5 gives the likely causes and a fix to make and test again.

**Limits of the testing:**
- Listeners are simulated (D38), so the feedback-learning results show the loop works, not what real listeners would gain.
- The load test measures one laptop, with one gateway process and one model-server process.
- The Postgres queries are covered by the stack tests and checks against the real database, not by unit tests.

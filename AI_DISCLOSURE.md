# AI disclosure

Required by the DA3408 code of conduct on fair use of AI. Updated as the project goes.

## 1. Tools used

- Claude Code (Anthropic)

## 2. How the tools were used

- Planning: reviewed the course materials and turned them into the project to-do list.
- Phase 0: drafted the setup steps and boilerplate (`.gitignore`, `.gitattributes`, pre-commit and ruff configuration, README, this file, the decision-log draft) and the first version of the batch generator (`src/spatify/batches.py`) and its tests.
- Phase 1: drafted the pipeline code (ingest, validate and features stages, the EDA and benchmark scripts), its tests and the setup steps.
- Phase 2: Claude Code wrote the model code (the recommender, its MLflow pyfunc packaging, the train, evaluate and register stages, the `MLproject` and the `python_env.yaml` export), its tests and decision-log entries D16–D22. It also ran the pipeline, the promotion-gate check and the serving check on my machine, which produced `metrics/evaluate.json`, `metrics/register.json`, `reports/evaluate/` and the `evidence/phase2_*` logs.
- Phase 3: Claude Code wrote the serving stack (the FastAPI gateway, the model-server supervisor, the Dockerfiles, `docker-compose.yml`, the `.env` generator, the smoke test), its tests and decision-log entries D23–D29. It also built the images, started the stack, trained its first champion and ran the smoke test and the rollback check on my machine, which produced the `evidence/phase3_*` logs. Later it reduced the stack's and the pipeline's memory use (D30, `scripts/peak_memory.py`), restarted the stack to apply the change, and measured the result.
- After Phase 3 (step 0.6 of my plan): Claude Code made the Ray stages' outputs identical across reruns on the same data (D31) and added a test for it. It ran the pipeline on my machine to check, which produced `evidence/phase1_deterministic_outputs.log` and registered model version 4 (not promoted).
- Phase 4: Claude Code wrote the frontend (the Streamlit Discover, operator console and Help pages, its gateway client and metrics), the gateway's pipeline view and "why" labels, the frontend's Dockerfile and Compose service, their tests and decision-log entries D32–D34. It ran the tests; it didn't build the images or start the stack.
- Phase 5A: Claude Code wrote the drift check (PSI and KS, the drift_check stage), the pipeline worker (its loop, DVC runner, Postgres records, metrics, Dockerfile and Compose service), the gateway's /drift and the console's drift and run views, their tests and decision-log entries D35–D36. It ran the tests, checked the worker's SQL against the running Postgres inside a transaction it rolled back, and ran the drift stage's code on the data into a scratch file; it didn't build the images, start the worker or run the pipeline.
- Phase 5A fix: when my first try failed (the worker image had no user account for uid 1000, and the worker retried the failure about once a second), Claude Code stopped the worker, fixed the image and the retry handling, added tests, rebuilt the worker image and checked it in a throwaway container. It didn't start the worker or run the pipeline.
- Phase 5B: Claude Code wrote the feedback learning (the logistic-regression feature weights, evaluation on held-out likes and skips and the feedback promotion gate, exploration slots, feedback drift, the worker's feedback trigger, snapshot export and redeploy timing), the simulated listeners and `scripts/simulate_users.py`, `scripts/force_bad_model.py` and `scripts/feedback_experiment.py`, the Discover, console and Help changes, their tests and decision-log entries D37–D39. It ran the tests and ran `scripts/feedback_experiment.py` on my machine, which produced `evidence/phase5_feedback_offline.log` and `evidence/phase5_feedback_offline_fisher.log`. After my step 5.4 run, it explained why the feedback model wasn't promoted and corrected D37.
- Phase 6: Claude Code wrote the monitoring:
  - the Prometheus scrape config, the recording and alert rules and their promtool unit tests;
  - the Alertmanager config, the Grafana provisioning, and the dashboard builder with the four dashboards it generates;
  - the Compose services;
  - the model server's and worker's new metrics, the gateway's `/alerts` endpoint and Summary metric, and the console's Monitoring tab;
  - the simulator's load options, their tests and decision-log entries D40–D42.

  It generated every PromQL query and alert rule (each is marked in the files, as in A4), and reused A4's two node-exporter regexes. It added `prometheus-flask-exporter` to the lock file and pulled the Alertmanager image. It ran the tests, and promtool and amtool in throwaway containers. It ran a throwaway MLflow server to check `--expose-prometheus` and a throwaway Grafana to check the dashboards load. It checked the alerts table's SQL against my running Postgres inside a transaction it rolled back. It didn't build the images, start the monitoring services or restart the stack. When my run showed the frontend's target down, it read Prometheus's history to find the cause, and changed the frontend so its metrics server starts with the process instead of on the first page view. When a batch I dropped in again was ignored, it found and fixed the worker's check for new batches. It checked the simulator against the gateway app run in-process with test fakes, and the bad-model script against a throwaway MLflow registry outside the repo. After the drive was remounted under a new path, it reinstalled the project's virtual environment so its tools would run again. It didn't build the images, start the stack or run the pipeline.
- Phase 7: Claude Code wrote:
  - the whole-pipeline integration test and the test that no key reaches the logs;
  - the load test (`scripts/load_test.py`, after the course's A5 pattern) and the failure test (`scripts/failure_test.py`);
  - the `Makefile` with `make ci` and its pipeline script, and the coverage settings;
  - `docs/security.md`, `docs/test_plan.md` and the `docs/test_report.md` template;
  - decision-log entries D43–D44.

  It ran the tests and a coverage run on my machine. It checked the load test against a mock gateway, and the failure test's kill command against throwaway local processes. That check showed the first version was too broad (it killed the shell running it), so it was narrowed to an exact match. It didn't run the load test, the failure test or `make ci` against the stack.
- Phase 7 report (step 7.4): Claude Code finished `docs/test_report.md` from my evidence files and screenshots. Two defects showed up in my runs:
  - The failure-run alert never fired. It found the cause in Prometheus's history and fixed it in the worker's metrics, with a test.
  - The 5xx panel's axis read 0-10000%. It fixed the dashboard builder, and checked the running Grafana had picked the change up.

  It recorded both in D41 and D42. It didn't run the load test, the failure test or the usability test.
- Phase 8 (documents): Claude Code drafted:
  - `docs/architecture.md`, `docs/hld.md`, `docs/lld.md` and `docs/user_manual.md`, and rewrote the README;
  - `scripts/export_api_docs.py`, which writes `docs/openapi.json` and the LLD's API reference from the code, and `tests/test_docs.py`.

  The HLD summarises the decisions in `docs/decision_log.md`; it adds no new design. To make the API specification complete, it added the error responses and field descriptions to the gateway's OpenAPI spec, and corrected the DVC remote's path in D7 and the in-app Help page's operator section. It ran the tests; it didn't run the stack.
- Phase 8 (finishing the documents): Claude Code put my screenshots into the user manual, renaming one file to the name the manual uses, and fitted the captions to what each screenshot shows. It checked the documents against the code and updated what was out of date: the test report with my usability results and the failure-test rerun, the test counts, the data card. The screenshots showed two display bugs in the console (the pipeline diagram's labels and squeezed charts); it fixed them, with tests. It audited the repo against the rubric and the problem statement's deliverables. It ran the tests; it didn't rebuild the images or run the stack.

What I did myself (check each line is true before submitting):

- Made the setup decisions recorded in `docs/decision_log.md` and reviewed every entry.
- Ran every setup command myself and read each generated file before committing it.

## 3. Impact

This tool helped me really shape up my project and assisted me with the coding part of the project. I used Claude Code to generate the files I couldn't create by myself, and debugged a lot of bugs in between. It really sped up the entire process and helped me finish it much earlier than I thought I would.

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
- Phase 5B: Claude Code wrote the feedback learning (the logistic-regression feature weights, evaluation on held-out likes and skips and the feedback promotion gate, exploration slots, feedback drift, the worker's feedback trigger, snapshot export and redeploy timing), the simulated listeners and `scripts/simulate_users.py`, `scripts/force_bad_model.py` and `scripts/feedback_experiment.py`, the Discover, console and Help changes, their tests and decision-log entries D37–D39. It ran the tests and ran `scripts/feedback_experiment.py` on my machine, which produced `evidence/phase5_feedback_offline.log` and `evidence/phase5_feedback_offline_fisher.log`. After my step 5.4 run, it explained why the feedback model wasn't promoted and corrected D37. It checked the simulator against the gateway app run in-process with test fakes, and the bad-model script against a throwaway MLflow registry outside the repo. After the drive was remounted under a new path, it reinstalled the project's virtual environment so its tools would run again. It didn't build the images, start the stack or run the pipeline.

What I did myself (check each line is true before submitting):

- Made the setup decisions recorded in `docs/decision_log.md` and reviewed every entry.
- Ran every setup command myself and read each generated file before committing it.

## 3. Impact

To be written at the end of the project: how the tool shaped the final submission.

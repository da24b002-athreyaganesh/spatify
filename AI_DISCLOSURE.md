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

What I did myself (check each line is true before submitting):

- Made the setup decisions recorded in `docs/decision_log.md` and reviewed every entry.
- Ran every setup command myself and read each generated file before committing it.

## 3. Impact

To be written at the end of the project: how the tool shaped the final submission.

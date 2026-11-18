# Spatify low-level design

The interfaces, data structures and algorithms of Spatify, in enough detail to call, change or re-implement any part. The parts are drawn in the [architecture](architecture.md); the reasons for each choice are in the [high-level design](hld.md) and the [decision log](decision_log.md).

Sections 3 and 4.2 are generated from the code by `scripts/export_api_docs.py`. Section 3 comes from the gateway's OpenAPI spec (the same spec `/docs` shows, saved as [openapi.json](openapi.json)); section 4.2 from the signature every packaged model is logged with. `tests/test_docs.py` fails if they fall out of date.

## 1. Code structure

All code is one Python package, `src/spatify`. Long-lived parts with state are classes: the recommender, the registry client, the stores, the worker, the model-server supervisor and the gateway's model client. Pipeline stages are functions: each stage module has pure functions over DataFrames and a `main()` entry point that DVC or MLflow runs. Every service reads its settings from environment variables.

| Package / module | Responsibility | Main classes and functions |
|---|---|---|
| `schema.py` | The raw columns, their types and valid ranges | `COLUMNS`, ranges |
| `batches.py` | The base catalogue and generated batches (`make_batches` stage) | `split_catalogue`, `shift_features`, `corrupt`, `build` |
| `pipeline/common.py` | Helpers for the stages | `load_params`, `init_ray`, `write_parquet`, `throughput` |
| `pipeline/ingest.py` | `ingest`: read every source file with Ray Data | `input_files`, `read_tracks` |
| `pipeline/validate.py` | `validate`: check rows; split valid and rejected | `check_rows`, `split`, `report` |
| `pipeline/features.py` | `features`: merge genres, engineer, scale, drift reference | `collapse_tracks`, `add_engineered`, `build` |
| `pipeline/drift.py` | `drift_check`: PSI and KS per batch and feature | `psi`, `batch_drift`, `report` |
| `model/data.py` | Loading the feature table; the held-out split | `load_tracks`, `eval_mask`, `eval_queries` |
| `model/recommender.py` | The recommender and Fisher weights | `Recommender`, `fisher_weights`, `QueryError` |
| `model/explain.py` | "Why" labels | `explain` |
| `model/pyfunc.py` | The model as MLflow packages it | `SpatifyModel`, `signature`, `log_candidate` |
| `model/feedback.py` | Learning from likes and skips; ranking metrics | `fit_weights`, `ranking_metrics`, `split_by_time` |
| `model/train.py`, `evaluate.py`, `register.py` | The model stages | `variants`, `build`, `score`, `should_promote`, `promote` |
| `model/tracking.py`, `plots.py` | Lineage tags; figures | `lineage`, `data_version`, `weights_figure` |
| `model_server.py` | The supervisor that follows the champion alias | `Supervisor`, `alias_version`, `tree_rss` |
| `gateway/app.py` | The REST API | `create_app`, `errors` |
| `gateway/schemas.py` | Request and response models | 26 Pydantic models (section 3) |
| `gateway/registry.py` | Aliases, versions, the champion's catalogue, rollback | `Registry` |
| `gateway/model_client.py` | Calls to the model server | `ModelClient` |
| `gateway/store.py` | The gateway's Postgres queries | `Store` |
| `gateway/pipeline_view.py` | The console's pipeline and drift views from DVC's files | `load_dag`, `drift_view` |
| `gateway/feedback_drift.py`, `metrics.py`, `logs.py`, `settings.py` | Feedback drift; metrics; JSON logs; settings | `feedback_drift`, `JsonFormatter`, `Settings` |
| `worker/worker.py` | What the worker does on each check | `Worker`, `Changes` |
| `worker/runner.py` | Running DVC and reading which stages ran | `run_dvc`, `follow`, `DvcResult` |
| `worker/store.py`, `metrics.py` | The worker's Postgres queries; metrics | `WorkerStore` |
| `frontend/` | The Streamlit app: `api.py` (gateway calls), `ui.py` (shared widgets and charts), `views/` (Discover, console, Help) | `GatewayError`, `Slider` |
| `simulation.py` | Simulated listeners | `Listener`, `new_query`, `rate_list`, `offline_lists` |
| `db_schema.py` | The Postgres tables (section 5) | `SCHEMA` |

## 2. REST conventions

- **Base address:** `http://localhost:8000` from the laptop, `http://gateway:8000` inside Compose. The frontend reads it from `API_BASE_URL`.
- **Format:** JSON request and response bodies. Times in ISO 8601, UTC. Interactive docs at `/docs`; the spec at `/openapi.json`.
- **Validation:** every body is a Pydantic model; request models reject unknown fields. A body or parameter that fails gets **422** with the failing field (`HTTPValidationError`).
- **Errors:** every other error has the body `{"detail": "..."}` ([Problem](#problem)). A 500 adds `request_id`. The codes each endpoint can return are listed with it in section 3.
- **Admin actions:** `POST /models/rollback` and `POST /pipeline/trigger` need the admin key in `X-API-Key`; Alertmanager's `POST /alerts` sends it as `Authorization: Bearer <key>`. A missing or wrong key gets **401**; with no `API_KEY_SHA256` set, admin actions get **503**.
- **Tracing:** every response carries `X-Request-ID` (the caller's, if it sent one), and every request is logged as one JSON line with that ID, the method, path, status and time.
- **Tags:** *listener* endpoints serve the Discover page; *operator* endpoints serve the console; *service* endpoints are for health checks, Prometheus and Alertmanager.

| Endpoint | Tag | Purpose |
|---|---|---|
| `POST /recommend` | listener | Top-K songs for a seed song, sliders or both |
| `POST /feedback` | listener | A like or skip on a song in a served list |
| `GET /tracks/search` | listener | Find seed songs by title or artist |
| `GET /models` | operator | Model versions, aliases, lineage, like rates |
| `POST /models/rollback` | operator | Make an earlier version the champion |
| `GET /pipeline/runs` | operator | Recent pipeline runs |
| `GET /pipeline/dag` | operator | The pipeline diagram with each stage's last result |
| `GET /pipeline/status` | operator | The latest run and counts per status |
| `POST /pipeline/trigger` | operator | Queue a retrain |
| `GET /drift` | operator | Each batch's drift, and feedback drift |
| `GET /alerts` | operator | Alerts, firing first |
| `POST /alerts` | service | Alertmanager's webhook |
| `GET /health` | service | 200 only when recommendations can be served |
| `GET /metrics` | service | Prometheus metrics (not in the spec) |

**Example: a recommendation and a like.**

```http
POST /recommend
{"track_id": "5SuOikwiRyPMVoIQDJUgSV", "sliders": {"energy": 0.8}, "k": 3}

200 OK
{"request_id": "0b7c…", "model_version": 8, "model_run_id": "f3a1…",
 "recommendations": [
   {"track_id": "…", "track_name": "…", "artists": "…", "album_name": "…", "genres": ["pop"],
    "popularity": 61, "rank": 1, "score": 0.97, "why": ["energetic", "danceable"], "explored": false},
   …]}

POST /feedback
{"request_id": "0b7c…", "track_id": "…", "action": "like"}

201 Created
{"status": "recorded", "model_version": 8}
```

## 3. Gateway API reference

<!-- BEGIN GENERATED: api (scripts/export_api_docs.py; don't edit by hand) -->

### `POST /recommend`

Top-k tracks for a seed track, slider values, or both. *Tag: listener.*

**Request body**: [RecommendRequest](#recommendrequest)

| Status | Meaning | Body |
|---|---|---|
| 200 | Successful Response | [RecommendResponse](#recommendresponse) |
| 400 | The query can't be answered: no seed and no slider this model uses | [Problem](#problem) |
| 404 | The seed track isn't in the served model's catalogue (see /tracks/search) | [Problem](#problem) |
| 503 | The model server is starting or switching versions (try again in a few seconds), or the feedback store is unavailable | [Problem](#problem) |
| 422 | Validation Error | [HTTPValidationError](#httpvalidationerror) |

### `POST /feedback`

A like or skip on one track of a list from /recommend. *Tag: listener.*

**Request body**: [FeedbackRequest](#feedbackrequest)

| Status | Meaning | Body |
|---|---|---|
| 201 | Successful Response | [FeedbackResponse](#feedbackresponse) |
| 400 | That track wasn't in this recommendation list | [Problem](#problem) |
| 404 | Unknown request_id: feedback must refer to a list from /recommend | [Problem](#problem) |
| 503 | The feedback store (Postgres) is unavailable | [Problem](#problem) |
| 422 | Validation Error | [HTTPValidationError](#httpvalidationerror) |

### `GET /tracks/search`

Seed tracks the served model knows, best title matches first. *Tag: listener.*

| Parameter | In | Type | Required | Limits | Description |
|---|---|---|---|---|---|
| `q` | query | string | yes | length ≥ 2; length ≤ 100 | Part of a title or artist |
| `limit` | query | integer | no | ≥ 1; ≤ 50; default `10` | Most tracks to return |

| Status | Meaning | Body |
|---|---|---|
| 200 | Successful Response | [SearchResponse](#searchresponse) |
| 503 | No model is being served yet, so there's nothing to search | [Problem](#problem) |
| 422 | Validation Error | [HTTPValidationError](#httpvalidationerror) |

### `GET /models`

Registered versions, their aliases, lineage and like rate. *Tag: operator.*

| Status | Meaning | Body |
|---|---|---|
| 200 | Successful Response | [ModelsResponse](#modelsresponse) |
| 503 | The model registry (MLflow) is unavailable | [Problem](#problem) |

### `POST /models/rollback`

Make an earlier version champion (default: `previous`). *Tag: operator.*

**Needs the admin key** in `X-API-Key`.

**Request body** (optional): [RollbackRequest](#rollbackrequest) or null

| Status | Meaning | Body |
|---|---|---|
| 200 | Successful Response | [RollbackResponse](#rollbackresponse) |
| 401 | Missing or wrong X-API-Key header | [Problem](#problem) |
| 409 | Impossible rollback: no champion, no `previous`, the version is already the champion, or it doesn't exist | [Problem](#problem) |
| 503 | The model registry is unavailable, or admin actions are disabled | [Problem](#problem) |
| 422 | Validation Error | [HTTPValidationError](#httpvalidationerror) |

### `POST /alerts`

Alertmanager's webhook: record each alert, firing or resolved (D41). *Tag: service.*

**Needs the admin key** in `Authorization: Bearer`.

**Request body**: [AlertmanagerWebhook](#alertmanagerwebhook)

| Status | Meaning | Body |
|---|---|---|
| 204 | Successful Response | none |
| 401 | Missing or wrong Bearer token | [Problem](#problem) |
| 503 | The feedback store (Postgres) is unavailable, or admin actions are disabled | [Problem](#problem) |
| 422 | Validation Error | [HTTPValidationError](#httpvalidationerror) |

### `GET /alerts`

Alerts from Prometheus (through Alertmanager): firing first, then the latest. *Tag: operator.*

| Parameter | In | Type | Required | Limits | Description |
|---|---|---|---|---|---|
| `limit` | query | integer | no | ≥ 1; ≤ 500; default `50` | Most alerts to return |

| Status | Meaning | Body |
|---|---|---|
| 200 | Successful Response | array of [AlertRecord](#alertrecord) |
| 503 | The feedback store (Postgres) is unavailable | [Problem](#problem) |
| 422 | Validation Error | [HTTPValidationError](#httpvalidationerror) |

### `GET /pipeline/runs`

Recent pipeline runs, newest first. *Tag: operator.*

| Parameter | In | Type | Required | Limits | Description |
|---|---|---|---|---|---|
| `limit` | query | integer | no | ≥ 1; ≤ 200; default `20` | Most runs to return |

| Status | Meaning | Body |
|---|---|---|
| 200 | Successful Response | array of [PipelineRun](#pipelinerun) |
| 503 | The feedback store (Postgres) is unavailable | [Problem](#problem) |
| 422 | Validation Error | [HTTPValidationError](#httpvalidationerror) |

### `GET /pipeline/dag`

The DVC pipeline: stages, data flow, each stage's last result and throughput. *Tag: operator.*

| Status | Meaning | Body |
|---|---|---|
| 200 | Successful Response | [PipelineDag](#pipelinedag) |
| 503 | The pipeline files (dvc.yaml, dvc.lock) aren't mounted | [Problem](#problem) |

### `GET /pipeline/status`

Pipeline Status *Tag: operator.*

| Status | Meaning | Body |
|---|---|---|
| 200 | Successful Response | [PipelineStatus](#pipelinestatus) |
| 503 | The feedback store (Postgres) is unavailable | [Problem](#problem) |

### `POST /pipeline/trigger`

Queue a pipeline run; the worker picks it up. *Tag: operator.*

**Needs the admin key** in `X-API-Key`.

**Request body** (optional): [TriggerRequest](#triggerrequest) or null

| Status | Meaning | Body |
|---|---|---|
| 202 | Successful Response | [PipelineRun](#pipelinerun) |
| 401 | Missing or wrong X-API-Key header | [Problem](#problem) |
| 503 | The feedback store (Postgres) is unavailable, or admin actions are disabled | [Problem](#problem) |
| 422 | Validation Error | [HTTPValidationError](#httpvalidationerror) |

### `GET /drift`

Each incoming batch's drift from the training data (PSI and KS), newest first, and feedback drift: the like rate of the latest likes and skips against before. *Tag: operator.*

| Status | Meaning | Body |
|---|---|---|
| 200 | Successful Response | [DriftReport](#driftreport) |
| 404 | No drift report yet: no batch checked and no feedback | [Problem](#problem) |

### `GET /health`

200 only when recommendations can be served: Postgres and the model server answer. *Tag: service.*

| Status | Meaning | Body |
|---|---|---|
| 200 | Successful Response | [Health](#health) |
| 503 | Postgres or the model server is down | [Health](#health) |

### Schemas

#### AlertRecord

| Field | Type | Required | Limits | Description |
|---|---|---|---|---|
| `fingerprint` | string | yes |  | Alertmanager's ID for the alert |
| `alertname` | string | yes |  | The rule's name, e.g. DataDrift |
| `status` | one of `"firing"`, `"resolved"` | yes |  | firing or resolved |
| `severity` | string or null | yes |  | warning or critical |
| `summary` | string or null | yes |  | One line |
| `description` | string or null | yes |  | What it means and what to do |
| `starts_at` | string (date-time) | yes |  | When it started firing |
| `ends_at` | string (date-time) or null | yes |  | Empty while firing |
| `received_at` | string (date-time) | yes |  | When the latest notice arrived |

#### AlertmanagerAlert

One alert in Alertmanager's webhook payload (the fields Spatify keeps).

| Field | Type | Required | Limits | Description |
|---|---|---|---|---|
| `status` | one of `"firing"`, `"resolved"` | yes |  |  |
| `labels` | object of string | yes |  |  |
| `annotations` | object of string | no |  |  |
| `startsAt` | string (date-time) | yes |  |  |
| `endsAt` | string (date-time) or null | no |  |  |
| `fingerprint` | string | yes | length ≥ 1; length ≤ 64 |  |

#### AlertmanagerWebhook

What Alertmanager posts to a webhook receiver; other fields are ignored.

| Field | Type | Required | Limits | Description |
|---|---|---|---|---|
| `alerts` | array of [AlertmanagerAlert](#alertmanageralert) | yes | items ≤ 500 |  |

#### DriftBatch

| Field | Type | Required | Limits | Description |
|---|---|---|---|---|
| `name` | string | yes |  | File name in data/incoming |
| `status` | string | yes |  | new, processed, quarantined, failed, removed or unknown |
| `first_seen` | string (date-time) or null | yes |  | When the worker first saw it |
| `message` | string or null | yes |  | What became of it |
| `rows` | integer or null | yes |  | Valid rows checked; none if validation rejected it |
| `drifted` | boolean or null | yes |  | Whether it drifted; empty if it wasn't checked |
| `features` | array of [DriftFeature](#driftfeature) | yes |  | One row per audio feature; empty if not checked |

#### DriftFeature

| Field | Type | Required | Limits | Description |
|---|---|---|---|---|
| `feature` | string | yes |  | Audio feature |
| `psi` | number | yes |  | Population Stability Index against the training data |
| `ks_stat` | number | yes |  | Kolmogorov-Smirnov statistic |
| `ks_pvalue` | number | yes |  | KS test p-value (reported; PSI decides) |
| `level` | string | yes |  | stable (< 0.1), moderate (0.1-0.2) or major (>= 0.2) |

#### DriftReport

| Field | Type | Required | Limits | Description |
|---|---|---|---|---|
| `reference_rows` | integer or null | yes |  | Training rows the batches are compared with |
| `thresholds` | object of number | yes |  | PSI levels: moderate and major |
| `batches` | array of [DriftBatch](#driftbatch) | yes |  | Newest first |
| `feedback` | [FeedbackDrift](#feedbackdrift) or null | no |  | The like rate: recent vs before |

#### FeedbackDrift

| Field | Type | Required | Limits | Description |
|---|---|---|---|---|
| `recent_events` | integer | yes |  | Ratings in the recent window |
| `recent_like_rate` | number or null | yes |  | Like rate of the latest ratings |
| `baseline_events` | integer | yes |  | Ratings in the window before |
| `baseline_like_rate` | number or null | yes |  | Like rate of the ratings before |
| `drop` | number or null | yes |  | baseline like rate minus recent like rate |
| `threshold` | number | yes |  | The drop that counts as drift |
| `enough_feedback` | boolean | yes |  | Whether both windows have enough ratings to judge |
| `drifted` | boolean | yes |  | True when the drop reaches the threshold |

#### FeedbackRequest

| Field | Type | Required | Limits | Description |
|---|---|---|---|---|
| `request_id` | string (uuid) | yes |  | From the /recommend answer the track was in |
| `track_id` | string | yes | length ≥ 1; length ≤ 64 | A track in that list |
| `action` | one of `"like"`, `"skip"` | yes |  | What the listener thought of it |

#### FeedbackResponse

| Field | Type | Required | Limits | Description |
|---|---|---|---|---|
| `status` | `"recorded"` | no | default `"recorded"` |  |
| `model_version` | integer or null | yes |  | The version that made the list |

#### HTTPValidationError

| Field | Type | Required | Limits | Description |
|---|---|---|---|---|
| `detail` | array of [ValidationError](#validationerror) | no |  |  |

#### Health

| Field | Type | Required | Limits | Description |
|---|---|---|---|---|
| `status` | one of `"ok"`, `"unavailable"` | yes |  | ok only when recommendations can be served |
| `checks` | object of string | yes |  | postgres, model_server and registry, each ok or what's wrong |

#### ModelVersion

| Field | Type | Required | Limits | Description |
|---|---|---|---|---|
| `version` | integer | yes |  | Registry version number |
| `aliases` | array of string | yes |  | champion (served), previous (rollback target), challenger |
| `run_id` | string | yes |  | MLflow run that trained it |
| `created_at` | string (date-time) | yes |  | When it was registered |
| `description` | string or null | yes |  | The promotion gate's verdict |
| `variant` | string or null | yes |  | Feature set and weighting, e.g. all-feedback |
| `dataset_version` | string or null | yes |  | DVC hash of the data it was trained on |
| `git_commit` | string or null | yes |  | Commit of the code that trained it |
| `trigger` | string or null | yes |  | What started its run: manual, new_data, drift, feedback, ci, forced_bad |
| `likes` | integer | no | default `0` | Likes on lists it served |
| `skips` | integer | no | default `0` | Skips on lists it served |
| `like_rate` | number or null | no |  | likes / (likes + skips) |

#### ModelsResponse

| Field | Type | Required | Limits | Description |
|---|---|---|---|---|
| `name` | string | yes |  | Registered model name |
| `champion` | integer or null | yes |  | The version listeners get |
| `previous` | integer or null | yes |  | The rollback target |
| `challenger` | integer or null | yes |  | The latest version the gate didn't promote |
| `versions` | array of [ModelVersion](#modelversion) | yes |  | Newest first |

#### PipelineDag

| Field | Type | Required | Limits | Description |
|---|---|---|---|---|
| `stages` | array of [StageView](#stageview) | yes |  | In dvc.yaml order |
| `inputs` | array of string | yes |  | Data the pipeline reads but no stage produces |
| `edges` | array of pair of string, string | yes |  | (from, to): stage or input to stage |

#### PipelineRun

| Field | Type | Required | Limits | Description |
|---|---|---|---|---|
| `id` | integer | yes |  | Run number |
| `requested_at` | string (date-time) | yes |  | When it was queued or the worker started it |
| `started_at` | string (date-time) or null | yes |  | Empty until the worker picks it up |
| `finished_at` | string (date-time) or null | yes |  | Empty while it runs |
| `trigger` | string | yes |  | manual, new_data, drift or feedback |
| `status` | string | yes |  | requested, running, succeeded or failed |
| `stages` | object | yes |  | Per stage: its result and seconds |
| `model_version` | integer or null | yes |  | The version this run registered, if any |
| `promoted` | boolean or null | yes |  | Whether that version became champion |
| `message` | string or null | yes |  | One-line explanation of what happened |

#### PipelineStatus

| Field | Type | Required | Limits | Description |
|---|---|---|---|---|
| `latest` | [PipelineRun](#pipelinerun) or null | yes |  | The newest run |
| `counts` | object of integer | yes |  | Runs per status |

#### Problem

The body of every error answer (4xx, 5xx); 422 adds the field that failed.

| Field | Type | Required | Limits | Description |
|---|---|---|---|---|
| `detail` | string | yes |  | What went wrong, in words a person can act on |
| `request_id` | string or null | no |  | Only on 500: quote it when reporting |

#### RecommendRequest

| Field | Type | Required | Limits | Description |
|---|---|---|---|---|
| `track_id` | string or null | no | length ≥ 1; length ≤ 64 | Seed track, from /tracks/search |
| `sliders` | [Sliders](#sliders) | no |  |  |
| `k` | integer | no | ≥ 1.0; ≤ 50.0; default `10` | Number of recommendations |

#### RecommendResponse

| Field | Type | Required | Limits | Description |
|---|---|---|---|---|
| `request_id` | string (uuid) | yes |  | Send this back with feedback on these tracks |
| `model_version` | integer or null | yes |  | Registry version that answered |
| `model_run_id` | string | yes |  | MLflow run that trained the answering model |
| `recommendations` | array of [Recommendation](#recommendation) | yes |  | Best match first |

#### Recommendation

| Field | Type | Required | Limits | Description |
|---|---|---|---|---|
| `track_id` | string | yes |  | Spotify track ID |
| `track_name` | string or null | yes |  | Title |
| `artists` | string or null | yes |  | Artists, separated by ';' |
| `album_name` | string or null | yes |  | Album |
| `genres` | array of string | yes |  | Every genre the track is listed under |
| `popularity` | integer | yes |  | Spotify popularity, 0-100 |
| `rank` | integer | yes |  | 1 is the closest match |
| `score` | number | yes |  | Cosine similarity to the query, -1 to 1 |
| `why` | array of string | no |  | Traits shared with the query that add most to the score, strongest first |
| `explored` | boolean | no | default `false` | A random pick in place of the lowest-ranked slot, to learn from |

#### RollbackRequest

| Field | Type | Required | Limits | Description |
|---|---|---|---|---|
| `version` | integer or null | no | ≥ 1.0 | Default: the `previous` alias |

#### RollbackResponse

| Field | Type | Required | Limits | Description |
|---|---|---|---|---|
| `champion` | integer | yes |  | The version now served |
| `previous` | integer | yes |  | The version it replaced (the next rollback's default) |
| `detail` | string | yes |  | What happens next, in words |

#### SearchResponse

| Field | Type | Required | Limits | Description |
|---|---|---|---|---|
| `query` | string | yes |  | The text searched for |
| `model_version` | integer | yes |  | The version whose catalogue was searched |
| `tracks` | array of [Track](#track) | yes |  | Title matches first, then artist matches |

#### Sliders

Audio features a listener can ask for directly, in the dataset's units. All optional.

| Field | Type | Required | Limits | Description |
|---|---|---|---|---|
| `danceability` | number or null | no | ≥ 0.0; ≤ 1.0 | 0 (not) to 1 (very) |
| `energy` | number or null | no | ≥ 0.0; ≤ 1.0 | 0 (calm) to 1 (intense) |
| `loudness` | number or null | no | ≥ -60.0; ≤ 5.0 | Average loudness, dB |
| `speechiness` | number or null | no | ≥ 0.0; ≤ 1.0 | 0 (music) to 1 (spoken word) |
| `acousticness` | number or null | no | ≥ 0.0; ≤ 1.0 | 0 (electric) to 1 (acoustic) |
| `instrumentalness` | number or null | no | ≥ 0.0; ≤ 1.0 | 1: no vocals |
| `liveness` | number or null | no | ≥ 0.0; ≤ 1.0 | 1: recorded live |
| `valence` | number or null | no | ≥ 0.0; ≤ 1.0 | 0 (sad) to 1 (happy) |
| `tempo` | number or null | no | ≥ 0.0; ≤ 250.0 | Beats per minute |
| `mode` | integer or null | no | ≥ 0.0; ≤ 1.0 | 0 minor, 1 major |

#### StageView

| Field | Type | Required | Limits | Description |
|---|---|---|---|---|
| `name` | string | yes |  | Stage name in dvc.yaml |
| `status` | string | yes |  | ok, failed, running or not run: the stage's last result |
| `command` | string | yes |  | The stage's command |
| `metrics` | object of number | yes |  | Rows, seconds, rows per second, ... |

#### Track

| Field | Type | Required | Limits | Description |
|---|---|---|---|---|
| `track_id` | string | yes |  | Spotify track ID |
| `track_name` | string or null | yes |  | Title |
| `artists` | string or null | yes |  | Artists, separated by ';' |
| `album_name` | string or null | yes |  | Album |
| `genres` | array of string | yes |  | Every genre the track is listed under |
| `popularity` | integer | yes |  | Spotify popularity, 0-100 |

#### TriggerRequest

| Field | Type | Required | Limits | Description |
|---|---|---|---|---|
| `trigger` | `"manual"` | no | default `"manual"` |  |

#### ValidationError

| Field | Type | Required | Limits | Description |
|---|---|---|---|---|
| `loc` | array of string or integer | yes |  |  |
| `msg` | string | yes |  |  |
| `type` | string | yes |  |  |
| `input` | any | no |  |  |
| `ctx` | object | no |  |  |

<!-- END GENERATED: api -->

## 4. Model server

### 4.1 Endpoints

The model server is `mlflow models serve` (MLflow's scoring server) for one registry version, started and restarted by a supervisor (`spatify.model_server`). It's reachable only inside Compose, at `http://model-server:8080`.

| Endpoint | Purpose |
|---|---|
| `POST /invocations` | Score queries (below) |
| `GET /ping`, `GET /health` | 200 once the model is loaded; the gateway probes `/ping` every 15 s |
| `GET /version` | The MLflow version |
| `:9103/metrics` | The supervisor's metrics (section 8) |

**Request** (MLflow's scoring protocol): one row per query, as `dataframe_records` (or `dataframe_split`), with K in `params`:

```json
{"dataframe_records": [{"track_id": "5SuOikwiRyPMVoIQDJUgSV", "energy": 0.8}], "params": {"k": 10}}
```

**Response:** `{"predictions": [...]}`, one object per recommended song, in rank order within each query.

**Errors:**
- An unknown seed song, or a row with no seed and no slider this model uses, gets **400** `{"error_code": "INVALID_PARAMETER_VALUE", "message": "..."}`. The gateway turns an unknown seed into its own 404, and the rest into 400.
- A wrongly typed column fails MLflow's schema enforcement, also 400.

**Behaviour:**
- **Scoring:** a query's features are scaled and weighted exactly as the catalogue was (section 7.1), then compared with all songs in one matrix product.
- **Results:** the seed and repeats of a song (same artists and title) are skipped. Unset sliders sit at the mean (z = 0), so they don't affect the ranking.
- **Lineage:** every answer carries `model_run_id`, the MLflow run that trained the model.

### 4.2 Contract (from the packaged model's signature)

<!-- BEGIN GENERATED: invocations (scripts/export_api_docs.py; don't edit by hand) -->

**Input columns** (one row per query; send `dataframe_records` or `dataframe_split`):

| Column | Type | Required |
|---|---|---|
| `track_id` | string | no |
| `danceability` | double | no |
| `energy` | double | no |
| `loudness` | double | no |
| `speechiness` | double | no |
| `acousticness` | double | no |
| `instrumentalness` | double | no |
| `liveness` | double | no |
| `valence` | double | no |
| `tempo` | double | no |
| `mode` | double | no |

**Parameters** (in `params`):

| Name | Type | Default |
|---|---|---|
| `k` | long | 10 |

**Output columns** (in `predictions`, one row per recommended track):

| Column | Type |
|---|---|
| `query` | long |
| `rank` | long |
| `track_id` | string |
| `track_name` | string |
| `artists` | string |
| `album_name` | string |
| `genres` | string |
| `popularity` | long |
| `score` | double |
| `model_run_id` | string |

<!-- END GENERATED: invocations -->

### 4.3 The supervisor

```
every POLL_SECONDS (10):
    wanted = the version the alias MODEL_ALIAS (champion) points to   # one REST call to MLflow
    if MLflow didn't answer: keep serving the current version
    elif there is no champion yet: wait
    elif wanted != serving or the server process died:
        stop the server (SIGTERM, then SIGKILL after 20 s)
        start `mlflow models serve -m models:/spatify/<wanted> --env-manager local --port 8080`
        in the background: time until /ping answers (spatify_model_server_ready_seconds)
```

### 4.4 The packaged model

What every registered version contains, so it answers without the training code or the pipeline:

| File | Contents |
|---|---|
| `artifacts/catalogue.parquet` | Every recommendable song: `track_id`, title, artists, album, genres (`;`-separated), popularity, and the raw values of the model's feature columns |
| `artifacts/model_info.json` | The feature columns, the scaler's means and standard deviations, the weights, the default K, the training run's ID |
| `code/spatify/` | This package, so the server imports `SpatifyModel` without installing it |
| `MLmodel`, `requirements.txt` | MLflow's descriptor, with the signature above; mlflow, numpy, pandas and pyarrow pinned |

## 5. Database (Postgres)

One Postgres 17 server holds two databases: `spatify`, the feedback store below, which the gateway and the worker create at start-up from `db_schema.py`; and `mlflow`, MLflow's own schema of runs and registry, managed by MLflow.

**`recommendations`**: every list the gateway served.

| Column | Type | Notes |
|---|---|---|
| `request_id` | uuid | Primary key; returned to the client |
| `created_at` | timestamptz | Default `now()` |
| `seed_track_id` | text | Null for slider-only queries |
| `sliders` | jsonb | The slider values set; default `{}` |
| `k` | integer | Songs asked for |
| `model_version` | integer | The registry version that answered |
| `model_run_id` | text | Its training run |
| `track_ids` | text[] | The list, in rank order |
| `explored` | text[] | The random "wildcard" picks in it; default `{}` |

**`feedback`**: every like and skip.

| Column | Type | Notes |
|---|---|---|
| `id` | bigserial | Primary key |
| `created_at` | timestamptz | Default `now()`; indexed |
| `request_id` | uuid | References `recommendations`; the list it was shown in |
| `track_id` | text | Must be in that list's `track_ids` (checked by the gateway) |
| `rank` | integer | Its position in the list (1 = top) |
| `action` | text | `like` or `skip` (CHECK) |
| `model_version`, `model_run_id` | integer, text | Copied from the list |
| | | UNIQUE (`request_id`, `track_id`): one rating per song per list; a second rating replaces the first (`ON CONFLICT … DO UPDATE`) |

**`pipeline_runs`**: the run queue and history.

| Column | Type | Notes |
|---|---|---|
| `id` | bigserial | Primary key |
| `requested_at`, `started_at`, `finished_at` | timestamptz | Queued (or started by the worker itself), claimed, finished |
| `trigger` | text | `manual`, `new_data`, `drift` or `feedback` |
| `status` | text | `requested`, `running`, `succeeded` or `failed` (CHECK) |
| `stages` | jsonb | Per stage: `{"status": "ok" \| "failed" \| "skipped", "seconds": …}`, plus `serving` after a promotion |
| `model_version` | integer | The version registered, if any |
| `promoted` | boolean | Whether it became champion |
| `message` | text | One-line explanation |

**`incoming_batches`**: every batch file the worker has seen.

| Column | Type | Notes |
|---|---|---|
| `name` | text | Primary key: the file name in `data/incoming` |
| `sha256` | text | Its contents' hash; a changed file counts as new |
| `first_seen` | timestamptz | Default `now()` |
| `status` | text | `new`, `processed`, `quarantined`, `failed` or `removed` (CHECK) |
| `run_id` | bigint | References `pipeline_runs`: the run that handled it |
| `drifted` | boolean | The drift check's verdict |
| `message` | text | What became of it |

**`alerts`**: every alert Alertmanager delivered.

| Column | Type | Notes |
|---|---|---|
| `fingerprint`, `starts_at` | text, timestamptz | Primary key together: one row per alert occurrence |
| `ends_at` | timestamptz | Set when resolved |
| `status` | text | `firing` or `resolved` (CHECK) |
| `alertname`, `severity`, `summary`, `description` | text | From the alert rule |
| `labels` | jsonb | All the alert's labels |
| `received_at` | timestamptz | When the latest notice arrived; default `now()` |

## 6. The pipeline

### 6.1 Stages (`dvc.yaml`)

Data stages run as plain Python; model stages run as MLflow Projects (`mlflow run . -e <stage> --env-manager=local --experiment-name spatify`), whose entry points are in `MLproject`. Every stage also depends on its own code modules (not listed).

| Stage | Command | Inputs | Outputs | Metrics | Params |
|---|---|---|---|---|---|
| `make_batches` | `python -m spatify.batches` | `data/raw/spotify_tracks.csv` | `data/base`, `data/batches` | | `batches` |
| `ingest` | `python -m spatify.pipeline.ingest` | `data/base`, `data/incoming` | `data/ingested` | `metrics/ingest.json` | |
| `validate` | `python -m spatify.pipeline.validate` | `data/ingested` | `data/validated` | `metrics/validate.json` | `validate` |
| `features` | `python -m spatify.pipeline.features` | `data/validated` | `data/features` | `metrics/features.json` | `features` |
| `drift_check` | `python -m spatify.pipeline.drift` | `data/validated` | | `metrics/drift.json` | `drift` |
| `eda` | `python scripts/eda.py` | the raw CSV | `reports/eda` | | |
| `python_env` | `python scripts/export_python_env.py` | `uv.lock` | `python_env.yaml` | | |
| `train` | `mlflow run . -e train` | `data/features`, `data/feedback`, `MLproject`, `python_env.yaml` | `models/candidates.json` | | `features.columns`, `split`, `train`, `feedback` |
| `evaluate` | `mlflow run . -e evaluate` | `data/features`, `data/feedback`, the candidates | `reports/evaluate/*.png` | `metrics/evaluate.json` | `split`, `evaluate`, `feedback` |
| `register` | `mlflow run . -e register` | `data/features`, `data/feedback`, the candidates, `metrics/evaluate.json` | | `metrics/register.json` | `split`, `evaluate`, `register`, `feedback` |

### 6.2 Files between stages

| Path | Format | Contents |
|---|---|---|
| `data/ingested/` | Parquet, named by position | Every row as text, plus `source` (its file name) |
| `data/validated/` | Parquet; `rejected.csv` | Typed valid rows; rejected rows with the reason and the offending value |
| `data/features/tracks/` | Parquet | One row per song: details, `genres`, raw features, `key_sin`, `key_cos` |
| `data/features/scaler.json` | JSON | Mean and standard deviation per feature column |
| `data/features/reference.parquet` | Parquet | A seeded sample of 10,000 songs' raw features (the drift reference) |
| `data/feedback/feedback.parquet` | Parquet | The feedback snapshot: `request_id`, `created_at`, `seed_track_id`, `sliders` (JSON text), `track_id`, `rank`, `action`, `explored`, `model_version` |
| `models/candidates.json` | JSON | The training run, its experiment and each candidate's variant, run ID and model URI |
| `metrics/*.json` | JSON | Each stage's numbers: rows, seconds, rows per second, rejects, PSI per batch and feature, every candidate's scores, the gate's decision |

### 6.3 Settings (`params.yaml`)

| Section | Settings |
|---|---|
| `batches` | Seed, held-out genres, base fraction, batch size and counts, the drift shift, the planted error rate |
| `validate` | Valid ranges, required columns, `max_reject_rate` (0.05 per file) |
| `features` | Feature columns, reference sample size, seed |
| `split` | Held-out fraction (0.1) and seed |
| `train` | K, feature sets, weightings |
| `evaluate` | K, number of held-out seed queries, seed |
| `register` | Model name, `min_gain` (0.005, genre match), `min_gain_feedback` (0.01, NDCG) |
| `feedback` | Snapshot path, held-out share (0.2), `min_events` (1,000), `min_test_lists` (30), k for NDCG (3), `retrain_every` (2,000) |
| `drift` | Reference source, features, bins (10), `psi_moderate` (0.1), `psi_major` (0.2), `min_moderate` (2) |

## 7. Algorithms

### 7.1 Scoring

For feature columns *j*, the scaler's mean μ and standard deviation σ, and weights w:

- z_j = (x_j − μ_j) / σ_j × w_j for the query and for every song.
- score = cos(z_query, z_song) = ẑ_query · ẑ_song (both scaled to length 1).
- The top K by score, skipping the seed and any song (artists + title) already listed. Exact, one matrix product; queries in chunks of 256.

Weightings:
- **Uniform:** w_j = 1.
- **Fisher:** w_j = √(between-genre variance / within-genre variance), from the training split only.
- **Feedback:** w_j = √max(0, −c_j), where c_j is the logistic-regression coefficient of (z_query,j − z_song,j)² in predicting a like.

All weights are scaled so their mean square is 1.

### 7.2 "Why" labels

Each feature's share of the score is ẑ_query,j × ẑ_song,j (exact, since the shares add up to the score). The labels name up to 3 features with the largest positive shares, each at least 5% of the score, by a fixed word per feature and direction ("energetic", "acoustic", "minor key").

### 7.3 Drift

- **PSI** per feature: the reference's deciles define 10 bins, and PSI = Σ_b (p_b − q_b) · ln(p_b / q_b), where p is the batch's share and q the reference's share in bin b.
- **Levels:** *stable* < 0.1 ≤ *moderate* < 0.2 ≤ *major*.
- **The rule:** a batch has drifted if any feature is major or at least two are moderate. The two-sample KS statistic and p-value are reported, not used.

### 7.4 The promotion gate

1. The best candidate is registered and given the `challenger` alias.
2. It is promoted if there is no champion, or if challenger − champion ≥ the minimum gain, both scored on the same data:
   - **Before feedback:** genre match@10 on 2,000 held-out seed songs (min gain 0.005).
   - **With at least 30 held-out feedback lists:** NDCG@3 over the latest 20% of lists (min gain 0.01).
3. A promotion sets `previous` to the old champion and `champion` to the new version, and removes `challenger`.

### 7.5 Feedback learning and drift

- **The snapshot:** every rating, split by time: the earlier 80% of lists train, the latest 20% are held out.
- **The feedback model:** fitted only with at least 1,000 usable ratings, both likes and skips.
- **Exploration:** ⌊0.1 × K⌋ slots (1 in a list of 10) are replaced, lowest rank first, by random catalogue songs, marked `explored`.
- **Feedback drift:** the like rate of the latest 200 ratings against the 1,000 before. Drift is a drop of 0.15 or more, once there are at least 200 recent and 500 earlier ratings.

## 8. The worker

```
every POLL_SECONDS (15):
    run = claim the oldest `requested` row in pipeline_runs
    if run:                                    # queued from the console or the API
        handle(run, changes in data/incoming, retrain=True)
    elif files in data/incoming are new, changed, back after removal, or gone:
        handle(new run "new_data", changes, retrain=False)
    elif ratings since the last snapshot >= feedback.retrain_every (2,000):
        handle(new run "feedback", no changes, retrain=True)

handle(run, changes, retrain):
    record the batches; dvc add data/incoming          # when files changed
    dvc repro drift_check                              # ingest, validate, drift_check
        validate failed  -> move the bad batches to data/quarantine; run failed; stop
        another failure  -> mark the batches failed; run failed; stop
    drifted = batches whose drift verdict is true
    if not (retrain or drifted): record "no drift"; stop
    if drifted and not retrain: trigger = "drift"
    export every rating to data/feedback/feedback.parquet; dvc add data/feedback
    dvc repro                                          # features, train, evaluate, register
    if training was up to date and a retrain was asked for: dvc repro --force --downstream train
    if register promoted a version: wait until the model server answers with it (≤ 3 min)
        and record retrain-to-redeploy time
    finally: record the run's status, stages, version, promotion, message and metrics
```

A worker that starts finds any run left `running` by a previous worker and marks it failed. After a failed run it waits a full poll before the next.

## 9. Metrics

Every service's `/metrics`, scraped every 5 s. Types: C counter, G gauge, H histogram, S summary, I info.

| Service (port) | Metric | Type | Labels |
|---|---|---|---|
| gateway (8000) | `spatify_gateway_requests_total` | C | endpoint, method, status |
| | `spatify_gateway_request_seconds` | H | endpoint |
| | `spatify_gateway_requests_in_progress` | G | |
| | `spatify_model_server_request_seconds`, `spatify_model_server_errors_total`, `spatify_model_server_up` | H, C, G | errors: kind |
| | `spatify_recommendations_total`, `spatify_explored_slots_total` | C | model_version |
| | `spatify_feedback_total` | C | action, model_version |
| | `spatify_feedback_rank` | S | action, model_version |
| | `spatify_feedback_like_rate`, `spatify_feedback_drift` | G | window |
| | `spatify_feedback_store_seconds`, `spatify_feedback_store_errors_total` | H, C | operation |
| | `spatify_champion_info` | I | version |
| | `spatify_alert_notifications_total` | C | alertname, status |
| worker (9102) | `spatify_pipeline_runs_total` | C | trigger, status |
| | `spatify_pipeline_run_seconds`, `spatify_pipeline_stage_seconds` | H | trigger; stage |
| | `spatify_pipeline_stage_rows_per_second`, `spatify_pipeline_stage_rows` | G | stage |
| | `spatify_pipeline_queue_length`, `spatify_pipeline_last_run_success`, `spatify_pipeline_last_run_timestamp_seconds` | G | |
| | `spatify_batches_quarantined_total` | C | |
| | `spatify_drift_psi`, `spatify_drift_detected` | G | feature |
| | `spatify_feedback_pending` | G | |
| | `spatify_retrain_to_redeploy_seconds` | H | |
| model-server (9103) | `spatify_model_server_serving_version`, `spatify_model_server_process_up`, `spatify_model_server_resident_memory_bytes` | G | |
| | `spatify_model_server_launches_total` | C | reason |
| | `spatify_model_server_ready_seconds` | H | |
| | `spatify_model_server_registry_errors_total` | C | |
| frontend (9101) | `spatify_frontend_page_views_total`, `spatify_frontend_feedback_clicks_total`, `spatify_frontend_errors_shown_total` | C | page; action; kind |
| | `spatify_frontend_gateway_calls_total`, `spatify_frontend_gateway_call_seconds` | C, H | endpoint, outcome |
| mlflow (5000) | `mlflow_http_request_total`, `mlflow_http_request_duration_seconds` | C, H | method, status |
| postgres-exporter, node-exporter | `pg_up`, `pg_stat_user_tables_n_live_tup`, `pg_database_size_bytes`; `node_cpu_seconds_total`, `node_memory_*`, `node_filesystem_*` | | |

The recording rules, alert rules and their tests are in `monitoring/prometheus/`; the dashboards are generated by `monitoring/grafana/build_dashboards.py`.

**Alertmanager's webhook** (`POST /alerts`) carries `{"alerts": [{"status", "labels", "annotations", "startsAt", "endsAt", "fingerprint"}, …]}`. The gateway stores one row per (fingerprint, startsAt), updated when the alert resolves.

## 10. Configuration

Each service reads environment variables, set in `docker-compose.yml` from `.env` (secrets) and fixed values.

| Service | Variables |
|---|---|
| gateway | `MODEL_SERVER_URL`, `MLFLOW_TRACKING_URI`, `DATABASE_URL`, `MODEL_NAME`, `API_KEY_SHA256`, `REFRESH_SECONDS` (15), `PIPELINE_DIR`, `EXPLORATION_RATE` (0.1), `MODEL_TIMEOUT_SECONDS`, `FEEDBACK_RECENT` (200), `FEEDBACK_BASELINE` (1,000), `FEEDBACK_DROP` (0.15), `LOG_LEVEL` |
| frontend | `API_BASE_URL`, `METRICS_PORT` (9101), and the tool links `MLFLOW_UI_URL`, `GRAFANA_URL`, `PROMETHEUS_URL`, `ALERTMANAGER_URL`, `RAY_DASHBOARD_URL`, `GATEWAY_DOCS_URL` |
| model-server | `MLFLOW_TRACKING_URI`, `MODEL_NAME`, `MODEL_ALIAS` (champion), `PORT` (8080), `WORKERS` (1), `POLL_SECONDS` (10), `METRICS_PORT` (9103) |
| worker | `DATABASE_URL`, `MLFLOW_TRACKING_URI`, `MODEL_SERVER_URL`, `REPO_DIR`, `POLL_SECONDS` (15), `METRICS_PORT` (9102), `SPATIFY_RAY_CPUS` (4), `SPATIFY_RAY_OBJECT_STORE_MB` (1024) |
| mlflow | `MLFLOW_SERVER_ENABLE_JOB_EXECUTION` (false); the command line sets the Postgres store, served artifacts, allowed hosts and `--expose-prometheus` |
| postgres | `POSTGRES_USER`, `POSTGRES_PASSWORD` (from `.env`) |
| alertmanager | `WEBHOOK_TOKEN` (the admin key, from `.env`) |
| grafana | `GF_*`: anonymous viewer, embedding allowed, no plugin downloads, the home dashboard |
| postgres-exporter | `DATA_SOURCE_NAME` |

`.env` holds `POSTGRES_USER`, `POSTGRES_PASSWORD`, `SPATIFY_API_KEY` and `API_KEY_SHA256`, generated by `scripts/make_env.py`; `.env.example` documents it.

## 11. Errors and logging

- **Gateway:** failures behind it become HTTP answers:

  | Failure | Status |
  |---|---|
  | Model server unreachable | 503, with "try again in a few seconds" |
  | A query the model rejects | 400, or 404 for an unknown seed |
  | Registry conflict | 409 |
  | MLflow unavailable | 503 |
  | Postgres unavailable | 503 |
  | Anything unexpected | 500, with the request ID; logged with its traceback |

- **Model server:** the supervisor keeps serving the current version if MLflow can't be reached, and restarts a crashed server.
- **Worker:** see section 8. Every exception becomes a failed run with a message; it never moves the champion and never retries at full speed.
- **Frontend:** every gateway failure becomes a plain message (`GatewayError`) saying what to do; no stack trace reaches the page.
- **Logs:** the gateway writes one JSON object per line (time, level, logger, message, request ID, method, path, status, time taken, model version), never headers or bodies (D44). The other services use Python's standard format. `docker compose logs <service>` shows them.

## 12. Sequences

**A recommendation**

```mermaid
sequenceDiagram
    actor L as Listener
    participant F as frontend
    participant G as gateway
    participant M as model-server
    participant P as postgres
    L->>F: pick a song or set sliders, press "Get recommendations"
    F->>G: POST /recommend {track_id, sliders, k}
    G->>G: validate (422 if bad)
    G->>M: POST /invocations {dataframe_records, params: {k}}
    M-->>G: predictions + model_run_id
    G->>G: run -> version; 1 wildcard; "why" labels
    G->>P: INSERT recommendations (request_id, version, track_ids, explored)
    G-->>F: 200 {request_id, model_version, recommendations}
    F-->>L: the list, with Like and Skip
    L->>F: Like
    F->>G: POST /feedback {request_id, track_id, action}
    G->>P: SELECT the list; INSERT feedback (rank, version)
    G-->>F: 201 {status: recorded}
```

**A drifted batch, retrained and promoted**

```mermaid
sequenceDiagram
    actor O as Operator
    participant W as worker
    participant D as DVC (in the repo)
    participant R as MLflow
    participant S as model-server
    participant P as postgres
    O->>D: copy a batch into data/incoming
    W->>D: dvc add data/incoming; dvc repro drift_check
    D-->>W: validate ok; drift: energy PSI 0.21, 3 moderate
    W->>P: record the run (trigger drift)
    W->>D: export feedback; dvc repro (features, train, evaluate, register)
    D->>R: candidate runs; register: challenger beats champion by ≥ min gain
    R-->>D: champion -> new version, previous -> old
    loop every 10 s
        S->>R: which version is champion?
    end
    S->>S: restart on the new version
    W->>S: POST /invocations until it answers with the new run
    W->>P: run succeeded, version promoted, retrain-to-redeploy seconds
```

**A rollback**

```mermaid
sequenceDiagram
    actor O as Operator
    participant F as frontend (console)
    participant G as gateway
    participant R as MLflow
    participant S as model-server
    O->>F: Models tab: pick a version, tick, "Roll back" (admin key)
    F->>G: POST /models/rollback {version}  X-API-Key
    G->>R: champion -> version; previous -> old champion; tag rolled_back_at
    G-->>F: 200 {champion, previous, detail}
    S->>R: alias moved (within 10 s)
    S->>S: restart on that version (about 5 s)
```

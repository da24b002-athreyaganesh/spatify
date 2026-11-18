# Security note

What Spatify protects, how, and what it leaves out. The course guidelines ask for "security measures (encryption) to protect data and models"; this says where encryption is used, where it isn't, and why (D44).

Spatify runs on one laptop for one operator. Every port is published on `127.0.0.1` only, so nothing is reachable from the network; the threats that matter are a leaked secret, an unauthorised admin action from the same machine, and bad input reaching the model or the database.

## Secrets

| Secret | Where it lives | Protection |
|---|---|---|
| Admin key (`SPATIFY_API_KEY`) | `.env` on the laptop only | Generated randomly by `scripts/make_env.py` (24 bytes, `secrets.token_urlsafe`). `.env` is created with permissions `0600` and is git-ignored; `.env.example` is committed instead. |
| The admin key, as the gateway knows it | `API_KEY_SHA256` in the gateway's environment | **Only its SHA-256 hash**: the gateway never holds the key. It hashes what a request sends and compares with `hmac.compare_digest` (constant time, so timing reveals nothing). |
| The admin key, for Alertmanager's webhook | `/tmp/webhook_token` inside the Alertmanager container | Written at start-up with `umask 077` (readable by that container's user only); not in the committed config. |
| Postgres password | `.env`; the Postgres, MLflow, gateway, worker and postgres-exporter containers | Random (16 bytes, hex); never logged. |
| Grafana admin password | `GRAFANA_ADMIN_PASSWORD` in `.env` (default `admin`) | Needed only to edit dashboards; viewing is read-only and anonymous. **Set it in `.env`** before sharing the laptop. |

**No secrets in logs.** The gateway logs a fixed list of fields per request (`request_id`, method, path, status, time, model version, alert labels), never headers or bodies, so a key sent as `X-API-Key` or `Authorization: Bearer` can't reach a log. `tests/test_gateway.py::test_the_admin_key_never_reaches_the_logs` sends the key right and wrong, as a header and as a Bearer token, and checks no log line contains it. The failure test reads the key from `.env` and never prints it.

**No secrets in the repository.** `git ls-files` lists `.env.example` but not `.env`; the DVC remote holds data and models, not secrets.

## Admin actions need the key

| Endpoint | What it does | Needs |
|---|---|---|
| `POST /models/rollback` | Changes the model listeners get | `X-API-Key` |
| `POST /pipeline/trigger` | Starts a retrain | `X-API-Key` |
| `POST /alerts` | Records alerts (Alertmanager's webhook) | `Authorization: Bearer` |

A missing or wrong key gets `401`; with no hash configured, admin actions are switched off (`503`). The console keeps the key the operator types in that browser session only; the frontend's container never holds it (D32). Everything else is read-only, or a listener action that can only refer to lists the gateway itself served (feedback must name a served list and one of its tracks).

## Input and data

- **Validation at the edge:** every request body is a Pydantic model with `extra="forbid"` and value ranges (sliders 0-1, tempo 0-250, k 1-50, IDs at most 64 characters). A bad request gets `422` naming the field, before anything reaches the model or the database.
- **No SQL injection:** every query uses psycopg parameters; no SQL is built from request values.
- **Incoming data** is validated row by row (schema, types, ranges) and a batch with more than 5% bad rows is quarantined (D10, D36), so malformed data can't reach training.
- **Models are loaded only from Spatify's own registry.** MLflow pyfunc models are pickled, and unpickling runs code; the model server loads only the `champion` alias of the local registry, which only the pipeline (and the forced-bad demo script) write.

## Containers

- Every service's own process runs as a non-root user:
  - the gateway, frontend, MLflow and model server as `app` (uid 10001);
  - the worker as the repo's owner (uid 1000, so its files stay the operator's);
  - Postgres's server as `postgres` (its entrypoint starts as root, then drops to that user);
  - the monitoring images as their own unprivileged users.
- Every container has a hard memory limit with no swap (D30), so one runaway service can't take the laptop down.
- node-exporter sees the host's filesystem **read-only** (`/:/host:ro`), as in A4.

## Encryption: what is and isn't encrypted

| | Encrypted? | Why |
|---|---|---|
| Secrets at rest | Hashed (the admin key, in the gateway); otherwise plain in a `0600` file | A hash is enough for checking a key; the other secrets must be readable by the services that use them. |
| Traffic between services | No (HTTP on the Compose network) | It never leaves the laptop; the Compose network is reachable only by Spatify's containers. |
| Traffic to the browser | No (HTTP on `127.0.0.1`) | Loopback traffic never reaches a network interface. |
| Data and models at rest | No (Docker volumes, the DVC cache and remote on disk) | Public data (the Spotify tracks dataset) and simulated feedback; no personal data. |

**If Spatify left the laptop**, these would change first: TLS on the gateway (`uvicorn --ssl-keyfile --ssl-certfile`) and Grafana (`GF_SERVER_PROTOCOL=https`), or one reverse proxy terminating TLS in front of both; full-disk encryption for the volumes (LUKS on Linux, BitLocker on this drive's Windows side); a secrets manager (or Docker secrets) instead of `.env`; real user accounts instead of one shared admin key; and Grafana's anonymous access switched off.

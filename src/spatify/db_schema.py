"""The Postgres tables, shared by the gateway and the worker (each creates any missing).

- recommendations, feedback: every served list and every like or skip on it (D26).
- pipeline_runs: the run queue and history; the gateway queues runs, the worker runs
  them and records their result (D29, D36).
- incoming_batches: every batch file the worker has seen in data/incoming, and what
  became of it (D36).
"""

SCHEMA = """
CREATE TABLE IF NOT EXISTS recommendations (
    request_id    uuid PRIMARY KEY,
    created_at    timestamptz NOT NULL DEFAULT now(),
    seed_track_id text,
    sliders       jsonb NOT NULL DEFAULT '{}',
    k             integer NOT NULL,
    model_version integer,
    model_run_id  text NOT NULL,
    track_ids     text[] NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    id            bigserial PRIMARY KEY,
    created_at    timestamptz NOT NULL DEFAULT now(),
    request_id    uuid NOT NULL REFERENCES recommendations (request_id),
    track_id      text NOT NULL,
    rank          integer NOT NULL,
    action        text NOT NULL CHECK (action IN ('like', 'skip')),
    model_version integer,
    model_run_id  text NOT NULL,
    UNIQUE (request_id, track_id)
);
CREATE INDEX IF NOT EXISTS feedback_created_at ON feedback (created_at);
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id            bigserial PRIMARY KEY,
    requested_at  timestamptz NOT NULL DEFAULT now(),
    started_at    timestamptz,
    finished_at   timestamptz,
    trigger       text NOT NULL,
    status        text NOT NULL DEFAULT 'requested'
                  CHECK (status IN ('requested', 'running', 'succeeded', 'failed')),
    stages        jsonb NOT NULL DEFAULT '{}',
    model_version integer,
    promoted      boolean,
    message       text
);
CREATE TABLE IF NOT EXISTS incoming_batches (
    name          text PRIMARY KEY,
    sha256        text NOT NULL,
    first_seen    timestamptz NOT NULL DEFAULT now(),
    status        text NOT NULL DEFAULT 'new'
                  CHECK (status IN ('new', 'processed', 'quarantined', 'failed', 'removed')),
    run_id        bigint REFERENCES pipeline_runs (id),
    drifted       boolean,
    message       text
);
"""

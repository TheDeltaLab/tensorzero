-- Modified by Delta-AI under Apache 2.0
-- Tag-based scheduled cleanup of the daily-partitioned payload tables.
-- Rules and run history live in Postgres so the dashboard can manage them;
-- the embedded gateway worker applies the rules (see `gateway.cleanup`).

-- A cleanup rule deletes payload rows older than `older_than_days` whose
-- inference tags contain `tag_key` (and equal `tag_value` when set).
CREATE TABLE tensorzero.cleanup_rules (
    id UUID PRIMARY KEY,
    tag_key TEXT NOT NULL CHECK (tag_key <> ''),
    -- NULL matches every row carrying `tag_key`, regardless of the tag value.
    tag_value TEXT,
    older_than_days INTEGER NOT NULL CHECK (older_than_days >= 1),
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- One row per cleanup pass (scheduled or manually triggered).
CREATE TABLE tensorzero.cleanup_runs (
    id UUID PRIMARY KEY,
    trigger TEXT NOT NULL CHECK (trigger IN ('schedule', 'manual')),
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    error TEXT
);

-- One row per (run, rule, target table): tracks progress so the dashboard can
-- show `rows_deleted / total_rows` for in-flight runs.
CREATE TABLE tensorzero.cleanup_run_steps (
    id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES tensorzero.cleanup_runs(id) ON DELETE CASCADE,
    rule_id UUID,
    table_name TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'done', 'failed')),
    total_rows BIGINT,
    rows_deleted BIGINT NOT NULL DEFAULT 0,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    error TEXT
);

CREATE INDEX idx_cleanup_run_steps_run_id ON tensorzero.cleanup_run_steps(run_id);

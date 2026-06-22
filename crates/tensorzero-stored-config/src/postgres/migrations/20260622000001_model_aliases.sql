-- Modified by Delta-AI under Apache 2.0
CREATE TABLE IF NOT EXISTS model_aliases (
    name TEXT PRIMARY KEY,
    task TEXT,  -- NULL = wildcard (matches all), otherwise "chat"|"embedding"|"rerank"
    targets JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE model_aliases IS 'Model aliases with optional task-type filtering.';

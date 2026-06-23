-- Modified by Delta-AI under Apache 2.0
CREATE TABLE IF NOT EXISTS tensorzero.model_aliases_configs (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    schema_revision INT NOT NULL,
    config JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMPTZ
);

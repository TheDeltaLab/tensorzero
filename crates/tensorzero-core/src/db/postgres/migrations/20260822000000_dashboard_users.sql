-- Modified by Delta-AI under Apache 2.0
-- Allowlist of Azure-authenticated emails that may access the dashboard.
CREATE TABLE IF NOT EXISTS tensorzero.dashboard_users (
    email TEXT PRIMARY KEY,
    is_admin BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by TEXT
);

CREATE INDEX IF NOT EXISTS dashboard_users_is_admin_idx
    ON tensorzero.dashboard_users (is_admin)
    WHERE is_admin;

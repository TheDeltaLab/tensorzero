-- Modified by Delta-AI under Apache 2.0
-- Imported Synapse API keys (`sk-syn-v1-…`) stored as bcrypt hashes.
-- TensorZero native keys remain in `tensorzero_auth_api_key` (SHA-256).

CREATE TABLE tensorzero_auth_synapse_api_key (
    "id" BIGSERIAL PRIMARY KEY,
    "organization" TEXT NOT NULL,
    "workspace" TEXT NOT NULL,
    "description" TEXT,
    "public_id" CHAR(12) NOT NULL,
    "bcrypt_hash" TEXT NOT NULL,
    "disabled_at" TIMESTAMPTZ NULL,
    "expires_at" TIMESTAMPTZ NULL,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "uniq_synapse_public_id" UNIQUE ("public_id")
);

-- Modified by Delta-AI under Apache 2.0
-- Add per-inference protection from retention cleanup.
--
-- `protected_at` on the inference metadata tables marks an inference as
-- protected. The nightly pg_cron cleanup archives protected rows (metadata +
-- payload) into permanent, non-partitioned archive tables before dropping old
-- partitions, so protected inferences survive retention forever.
--
-- Archive tables intentionally mirror the column order of their source tables
-- so `INSERT INTO ... SELECT *` / `SELECT d.*` stay valid.

-- ============================================================================
-- protected_at flag on inference metadata tables
-- ============================================================================

ALTER TABLE tensorzero.chat_inferences ADD COLUMN protected_at TIMESTAMPTZ;
ALTER TABLE tensorzero.json_inferences ADD COLUMN protected_at TIMESTAMPTZ;

-- ============================================================================
-- Archive tables (non-partitioned, never dropped by retention)
-- ============================================================================

CREATE TABLE tensorzero.chat_inferences_archive (
    id UUID NOT NULL,
    function_name TEXT NOT NULL,
    variant_name TEXT NOT NULL,
    episode_id UUID NOT NULL,
    processing_time_ms INTEGER,
    ttft_ms INTEGER,
    tags JSONB NOT NULL DEFAULT '{}',
    snapshot_hash BYTEA,
    created_at TIMESTAMPTZ NOT NULL,
    protected_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (id, created_at)
);

CREATE TABLE tensorzero.chat_inference_data_archive (
    id UUID NOT NULL,
    input JSONB NOT NULL,
    output JSONB NOT NULL,
    inference_params JSONB NOT NULL,
    extra_body JSONB NOT NULL DEFAULT '[]',
    dynamic_tools JSONB NOT NULL DEFAULT '[]',
    dynamic_provider_tools JSONB NOT NULL DEFAULT '[]',
    allowed_tools JSONB,
    tool_choice JSONB,
    parallel_tool_calls BOOLEAN,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (id, created_at)
);

CREATE TABLE tensorzero.json_inferences_archive (
    id UUID NOT NULL,
    function_name TEXT NOT NULL,
    variant_name TEXT NOT NULL,
    episode_id UUID NOT NULL,
    processing_time_ms INTEGER,
    ttft_ms INTEGER,
    tags JSONB NOT NULL DEFAULT '{}',
    snapshot_hash BYTEA,
    created_at TIMESTAMPTZ NOT NULL,
    protected_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (id, created_at)
);

CREATE TABLE tensorzero.json_inference_data_archive (
    id UUID NOT NULL,
    input JSONB NOT NULL,
    output JSONB NOT NULL,
    output_schema JSONB NOT NULL,
    inference_params JSONB NOT NULL,
    extra_body JSONB NOT NULL DEFAULT '[]',
    auxiliary_content JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (id, created_at)
);

-- ============================================================================
-- Archive-and-drop cleanup functions
-- ============================================================================
-- Both functions process one old partition per statement. Only cold partitions
-- are touched, so online writes to current partitions are never blocked. A
-- short lock_timeout makes the job fail fast (and retry the next night)
-- instead of queueing behind a lock on a busy database.

-- Archives protected rows from old monthly metadata partitions of
-- chat_inferences / json_inferences, then drops the partitions.
CREATE OR REPLACE FUNCTION tensorzero.archive_and_drop_old_metadata_partitions()
RETURNS void AS $$
DECLARE
    retention_days INT;
    cutoff_date DATE;
    table_record RECORD;
    partition_record RECORD;
    pattern TEXT;
    partition_month_start DATE;
BEGIN
    PERFORM set_config('lock_timeout', '5s', true);

    SELECT value::INT INTO retention_days
    FROM tensorzero.retention_config
    WHERE key = 'inference_metadata_retention_days';

    IF retention_days IS NULL THEN
        RAISE NOTICE 'inference_metadata_retention_days not configured, skipping protected-aware metadata cleanup';
        RETURN;
    END IF;

    cutoff_date := CURRENT_DATE - retention_days;

    FOR table_record IN
        SELECT * FROM (VALUES ('chat_inferences'), ('json_inferences')) AS t(table_name)
    LOOP
        pattern := '^' || table_record.table_name || '_\d{4}_\d{2}$';
        FOR partition_record IN
            SELECT tablename
            FROM pg_tables
            WHERE schemaname = 'tensorzero' AND tablename ~ pattern
        LOOP
            partition_month_start := to_date(substring(partition_record.tablename from '\d{4}_\d{2}$'), 'YYYY_MM');
            -- Drop only when the entire month is before the cutoff
            IF partition_month_start + INTERVAL '1 month' <= cutoff_date THEN
                EXECUTE format(
                    'INSERT INTO tensorzero.%I_archive SELECT * FROM tensorzero.%I WHERE protected_at IS NOT NULL',
                    table_record.table_name,
                    partition_record.tablename
                );
                EXECUTE format('DROP TABLE tensorzero.%I', partition_record.tablename);
            END IF;
        END LOOP;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Archives protected rows from old daily data partitions of
-- chat_inference_data / json_inference_data, then drops the partitions.
-- A data row is protected when its matching metadata row (in the live table or
-- already archived) has protected_at set. The created_at range predicate keeps
-- partition pruning effective on the metadata side.
CREATE OR REPLACE FUNCTION tensorzero.archive_and_drop_old_data_partitions()
RETURNS void AS $$
DECLARE
    retention_days INT;
    cutoff_date DATE;
    table_record RECORD;
    partition_record RECORD;
    pattern TEXT;
    partition_date DATE;
    metadata_table TEXT;
BEGIN
    PERFORM set_config('lock_timeout', '5s', true);

    SELECT value::INT INTO retention_days
    FROM tensorzero.retention_config
    WHERE key = 'inference_data_retention_days';

    IF retention_days IS NULL THEN
        RAISE NOTICE 'inference_data_retention_days not configured, skipping protected-aware data cleanup';
        RETURN;
    END IF;

    cutoff_date := CURRENT_DATE - retention_days;

    FOR table_record IN
        SELECT * FROM (VALUES
            ('chat_inference_data', 'chat_inferences'),
            ('json_inference_data', 'json_inferences')
        ) AS t(table_name, metadata_table)
    LOOP
        pattern := '^' || table_record.table_name || '_\d{4}_\d{2}_\d{2}$';
        FOR partition_record IN
            SELECT tablename
            FROM pg_tables
            WHERE schemaname = 'tensorzero' AND tablename ~ pattern
        LOOP
            partition_date := to_date(substring(partition_record.tablename from '\d{4}_\d{2}_\d{2}$'), 'YYYY_MM_DD');
            IF partition_date < cutoff_date THEN
                EXECUTE format(
                    'INSERT INTO tensorzero.%I_archive
                     SELECT d.* FROM tensorzero.%I d
                     WHERE EXISTS (
                         SELECT 1 FROM (
                             SELECT id, created_at, protected_at FROM tensorzero.%I
                             UNION ALL
                             SELECT id, created_at, protected_at FROM tensorzero.%I_archive
                         ) m
                         WHERE m.id = d.id AND m.created_at = d.created_at
                           AND m.protected_at IS NOT NULL
                           AND m.created_at >= %L AND m.created_at < %L
                     )',
                    table_record.table_name,
                    partition_record.tablename,
                    table_record.metadata_table,
                    table_record.metadata_table,
                    partition_date,
                    partition_date + 1
                );
                EXECUTE format('DROP TABLE tensorzero.%I', partition_record.tablename);
            END IF;
        END LOOP;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

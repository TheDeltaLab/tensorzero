// Modified by Delta-AI under Apache 2.0
//! Postgres queries for inference storage stats and retention configuration.
//!
//! Storage stats are catalog-only queries (`pg_total_relation_size`,
//! `pg_inherits`, `pg_class.reltuples`): they never scan the inference
//! business tables, which can be very large.

use crate::error::{Error, ErrorDetails};

use super::PostgresConnectionInfo;

/// Per-table storage statistics for an inference-related Postgres table.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct InferenceTableStorageStatsRow {
    pub name: String,
    pub total_bytes: i64,
    pub partition_count: i64,
    pub estimated_rows: i64,
}

/// Retention configuration stored in `tensorzero.retention_config`.
/// `None` means the key is absent: the corresponding data is kept forever.
#[derive(Debug, Clone, Default)]
pub struct InferenceRetentionRow {
    pub metadata_retention_days: Option<u32>,
    pub data_retention_days: Option<u32>,
}

const METADATA_RETENTION_KEY: &str = "inference_metadata_retention_days";
const DATA_RETENTION_KEY: &str = "inference_data_retention_days";

impl PostgresConnectionInfo {
    /// Returns per-table storage stats for the inference metadata/data tables
    /// and their archive tables.
    ///
    /// `total_bytes` uses `pg_total_relation_size` over the parent plus all
    /// of its partitions. `estimated_rows` sums the planner's `reltuples`
    /// estimate over the parent and its partitions.
    pub async fn get_inference_storage_stats(
        &self,
    ) -> Result<Vec<InferenceTableStorageStatsRow>, Error> {
        let pool = self.get_pool_result().map_err(|e| e.log())?;

        let rows = sqlx::query_as!(
            InferenceTableStorageStatsRow,
            r#"
            WITH tables AS (
                SELECT * FROM (VALUES
                    ('chat_inferences'),
                    ('json_inferences'),
                    ('model_inferences'),
                    ('chat_inference_data'),
                    ('json_inference_data'),
                    ('model_inference_data'),
                    ('chat_inferences_archive'),
                    ('chat_inference_data_archive'),
                    ('json_inferences_archive'),
                    ('json_inference_data_archive')
                ) AS t(name)
            ),
            parents AS (
                SELECT t.name, c.oid
                FROM tables t
                JOIN pg_namespace n ON n.nspname = 'tensorzero'
                JOIN pg_class c ON c.relname = t.name AND c.relnamespace = n.oid
            )
            SELECT
                p.name AS "name!",
                -- `pg_total_relation_size` on a partitioned parent only counts
                -- the (empty) parent itself, so sum the partitions explicitly.
                (
                    pg_total_relation_size(p.oid)
                    + COALESCE((
                        SELECT SUM(pg_total_relation_size(i.inhrelid))
                        FROM pg_inherits i
                        WHERE i.inhparent = p.oid
                    ), 0)
                )::BIGINT AS "total_bytes!",
                (SELECT COUNT(*) FROM pg_inherits WHERE inhparent = p.oid)::BIGINT AS "partition_count!",
                -- `reltuples` is the planner estimate (-1 = never analyzed).
                GREATEST(COALESCE((
                    SELECT SUM(c.reltuples)::BIGINT
                    FROM pg_class c
                    WHERE c.oid = p.oid
                       OR c.oid IN (SELECT inhrelid FROM pg_inherits WHERE inhparent = p.oid)
                ), 0), 0) AS "estimated_rows!"
            FROM parents p
            ORDER BY p.name
            "#
        )
        .fetch_all(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to query inference storage stats: {e}"),
            })
        })?;

        Ok(rows)
    }

    /// Reads the inference retention configuration from
    /// `tensorzero.retention_config`.
    pub async fn get_inference_retention(&self) -> Result<InferenceRetentionRow, Error> {
        let pool = self.get_pool_result().map_err(|e| e.log())?;

        let rows = sqlx::query!(
            r"
            SELECT key, value
            FROM tensorzero.retention_config
            WHERE key IN ('inference_metadata_retention_days', 'inference_data_retention_days')
            "
        )
        .fetch_all(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to read inference retention config: {e}"),
            })
        })?;

        let mut retention = InferenceRetentionRow::default();
        for row in rows {
            let days = row.value.parse::<u32>().map_err(|e| {
                Error::new(ErrorDetails::PostgresQuery {
                    message: format!(
                        "Invalid value `{}` for retention config key `{}`: {e}",
                        row.value, row.key
                    ),
                })
            })?;
            match row.key.as_str() {
                METADATA_RETENTION_KEY => retention.metadata_retention_days = Some(days),
                DATA_RETENTION_KEY => retention.data_retention_days = Some(days),
                _ => {}
            }
        }

        Ok(retention)
    }

    /// Writes the inference retention configuration to
    /// `tensorzero.retention_config`. A `Some` value upserts the key; a `None`
    /// value deletes the key (keep forever).
    pub async fn set_inference_retention(
        &self,
        metadata_retention_days: Option<u32>,
        data_retention_days: Option<u32>,
    ) -> Result<(), Error> {
        let pool = self.get_pool_result().map_err(|e| e.log())?;

        Self::set_retention_key(pool, METADATA_RETENTION_KEY, metadata_retention_days).await?;
        Self::set_retention_key(pool, DATA_RETENTION_KEY, data_retention_days).await?;

        Ok(())
    }

    async fn set_retention_key(
        pool: &sqlx::PgPool,
        key: &str,
        value: Option<u32>,
    ) -> Result<(), Error> {
        match value {
            Some(days) => {
                sqlx::query!(
                    r"
                    INSERT INTO tensorzero.retention_config (key, value, updated_at)
                    VALUES ($1, $2, NOW())
                    ON CONFLICT (key) DO UPDATE SET value = $2, updated_at = NOW()
                    ",
                    key,
                    days.to_string(),
                )
                .execute(pool)
                .await
                .map_err(|e| {
                    Error::new(ErrorDetails::PostgresQuery {
                        message: format!("Failed to write `{key}` config: {e}"),
                    })
                })?;
            }
            None => {
                sqlx::query!(
                    "DELETE FROM tensorzero.retention_config WHERE key = $1",
                    key,
                )
                .execute(pool)
                .await
                .map_err(|e| {
                    Error::new(ErrorDetails::PostgresQuery {
                        message: format!("Failed to delete `{key}` config: {e}"),
                    })
                })?;
            }
        }
        Ok(())
    }
}

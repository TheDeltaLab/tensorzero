// Modified by Delta-AI under Apache 2.0
//! Postgres queries for tag-based scheduled cleanup of payload tables.
//!
//! Only the five daily-partitioned `*_data` payload tables are ever deleted
//! from; the metadata tables (`chat_inferences`, `json_inferences`,
//! `model_inferences`, `batch_model_inferences`, `batch_requests`) are always
//! preserved. Every search and delete carries two upper bounds on
//! `created_at`: the rule's age cutoff (computed once in Rust and shared by
//! the search and delete statements) and `date_trunc('day', now())`, which
//! keeps the current day's active partition untouched and enables partition
//! pruning. The chat/json paths additionally skip inferences with
//! `protected_at` set (the batch tables have no such column).

use std::time::Duration;

use chrono::{DateTime, Utc};
use sqlx::PgPool;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::error::{Error, ErrorDetails};

/// Number of rows deleted per `DELETE` statement during a cleanup pass.
/// Small chunks keep each transaction short so cleanup does not hold locks
/// or I/O on a partition for long.
pub const CLEANUP_DELETE_CHUNK_SIZE: usize = 1000;

/// Pause between chunk deletes, throttling the load cleanup puts on Postgres.
pub const CLEANUP_CHUNK_DELAY: Duration = Duration::from_millis(100);

pub const CLEANUP_TRIGGER_SCHEDULE: &str = "schedule";
pub const CLEANUP_TRIGGER_MANUAL: &str = "manual";

pub const CLEANUP_RUN_STATUS_RUNNING: &str = "running";
pub const CLEANUP_RUN_STATUS_COMPLETED: &str = "completed";
pub const CLEANUP_RUN_STATUS_FAILED: &str = "failed";

pub const CLEANUP_STEP_STATUS_RUNNING: &str = "running";
pub const CLEANUP_STEP_STATUS_DONE: &str = "done";
pub const CLEANUP_STEP_STATUS_FAILED: &str = "failed";

/// A tag-based cleanup rule from `tensorzero.cleanup_rules`.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct CleanupRuleRow {
    pub id: Uuid,
    pub tag_key: String,
    pub tag_value: Option<String>,
    pub older_than_days: i32,
    pub enabled: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// A cleanup pass from `tensorzero.cleanup_runs`.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct CleanupRunRow {
    pub id: Uuid,
    pub trigger: String,
    pub status: String,
    pub started_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
    pub error: Option<String>,
}

/// Per-(run, rule, table) progress from `tensorzero.cleanup_run_steps`.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct CleanupRunStepRow {
    pub id: Uuid,
    pub run_id: Uuid,
    pub rule_id: Option<Uuid>,
    pub table_name: String,
    pub status: String,
    pub total_rows: Option<i64>,
    pub rows_deleted: i64,
    pub started_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
    pub error: Option<String>,
}

/// A daily-partitioned payload table targeted by tag-based cleanup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CleanupTargetTable {
    ChatInferenceData,
    JsonInferenceData,
    ModelInferenceData,
    BatchModelInferenceData,
    BatchRequestData,
}

impl CleanupTargetTable {
    /// All target tables, in cleanup order.
    pub const ALL: [Self; 5] = [
        Self::ChatInferenceData,
        Self::JsonInferenceData,
        Self::ModelInferenceData,
        Self::BatchModelInferenceData,
        Self::BatchRequestData,
    ];

    /// Unqualified table name, recorded on `cleanup_run_steps.table_name`.
    pub fn name(self) -> &'static str {
        match self {
            Self::ChatInferenceData => "chat_inference_data",
            Self::JsonInferenceData => "json_inference_data",
            Self::ModelInferenceData => "model_inference_data",
            Self::BatchModelInferenceData => "batch_model_inference_data",
            Self::BatchRequestData => "batch_request_data",
        }
    }
}

// --- Rule CRUD ---

pub async fn list_cleanup_rules(pool: &PgPool) -> Result<Vec<CleanupRuleRow>, Error> {
    sqlx::query_as!(
        CleanupRuleRow,
        r#"
        SELECT id, tag_key, tag_value, older_than_days, enabled, created_at, updated_at
        FROM tensorzero.cleanup_rules
        ORDER BY created_at
        "#
    )
    .fetch_all(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to list cleanup rules: {e}"),
        })
    })
}

pub async fn list_enabled_cleanup_rules(pool: &PgPool) -> Result<Vec<CleanupRuleRow>, Error> {
    sqlx::query_as!(
        CleanupRuleRow,
        r#"
        SELECT id, tag_key, tag_value, older_than_days, enabled, created_at, updated_at
        FROM tensorzero.cleanup_rules
        WHERE enabled
        ORDER BY created_at
        "#
    )
    .fetch_all(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to list enabled cleanup rules: {e}"),
        })
    })
}

pub async fn create_cleanup_rule(
    pool: &PgPool,
    tag_key: &str,
    tag_value: Option<&str>,
    older_than_days: i32,
    enabled: bool,
) -> Result<CleanupRuleRow, Error> {
    let id = Uuid::now_v7();
    sqlx::query_as!(
        CleanupRuleRow,
        r#"
        INSERT INTO tensorzero.cleanup_rules (id, tag_key, tag_value, older_than_days, enabled)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING id, tag_key, tag_value, older_than_days, enabled, created_at, updated_at
        "#,
        id,
        tag_key,
        tag_value,
        older_than_days,
        enabled,
    )
    .fetch_one(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to create cleanup rule: {e}"),
        })
    })
}

/// Updates all fields of a rule (full-replace semantics). Returns `None` when
/// no rule with `id` exists.
pub async fn update_cleanup_rule(
    pool: &PgPool,
    id: Uuid,
    tag_key: &str,
    tag_value: Option<&str>,
    older_than_days: i32,
    enabled: bool,
) -> Result<Option<CleanupRuleRow>, Error> {
    sqlx::query_as!(
        CleanupRuleRow,
        r#"
        UPDATE tensorzero.cleanup_rules
        SET tag_key = $2, tag_value = $3, older_than_days = $4, enabled = $5, updated_at = now()
        WHERE id = $1
        RETURNING id, tag_key, tag_value, older_than_days, enabled, created_at, updated_at
        "#,
        id,
        tag_key,
        tag_value,
        older_than_days,
        enabled,
    )
    .fetch_optional(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to update cleanup rule `{id}`: {e}"),
        })
    })
}

/// Deletes a rule. Returns `false` when no rule with `id` exists.
pub async fn delete_cleanup_rule(pool: &PgPool, id: Uuid) -> Result<bool, Error> {
    let result = sqlx::query!("DELETE FROM tensorzero.cleanup_rules WHERE id = $1", id)
        .execute(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to delete cleanup rule `{id}`: {e}"),
            })
        })?;
    Ok(result.rows_affected() > 0)
}

// --- Run / step bookkeeping ---

pub async fn start_cleanup_run(pool: &PgPool, trigger: &str) -> Result<CleanupRunRow, Error> {
    let id = Uuid::now_v7();
    sqlx::query_as!(
        CleanupRunRow,
        r#"
        INSERT INTO tensorzero.cleanup_runs (id, trigger, status)
        VALUES ($1, $2, 'running')
        RETURNING id, trigger, status, started_at, finished_at, error
        "#,
        id,
        trigger,
    )
    .fetch_one(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to start cleanup run: {e}"),
        })
    })
}

pub async fn finish_cleanup_run(
    pool: &PgPool,
    run_id: Uuid,
    status: &str,
    error: Option<&str>,
) -> Result<(), Error> {
    sqlx::query!(
        r#"
        UPDATE tensorzero.cleanup_runs
        SET status = $2, error = $3, finished_at = now()
        WHERE id = $1
        "#,
        run_id,
        status,
        error,
    )
    .execute(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to finish cleanup run `{run_id}`: {e}"),
        })
    })?;
    Ok(())
}

async fn start_cleanup_step(
    pool: &PgPool,
    run_id: Uuid,
    rule_id: Uuid,
    table: CleanupTargetTable,
    total_rows: i64,
) -> Result<Uuid, Error> {
    let id = Uuid::now_v7();
    sqlx::query_scalar!(
        r#"
        INSERT INTO tensorzero.cleanup_run_steps
            (id, run_id, rule_id, table_name, status, total_rows)
        VALUES ($1, $2, $3, $4, 'running', $5)
        RETURNING id
        "#,
        id,
        run_id,
        rule_id,
        table.name(),
        total_rows,
    )
    .fetch_one(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to start cleanup step for `{}`: {e}", table.name()),
        })
    })
}

async fn update_cleanup_step_progress(
    pool: &PgPool,
    step_id: Uuid,
    rows_deleted: i64,
) -> Result<(), Error> {
    sqlx::query!(
        "UPDATE tensorzero.cleanup_run_steps SET rows_deleted = $2 WHERE id = $1",
        step_id,
        rows_deleted,
    )
    .execute(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to update cleanup step `{step_id}` progress: {e}"),
        })
    })?;
    Ok(())
}

async fn finish_cleanup_step(
    pool: &PgPool,
    step_id: Uuid,
    status: &str,
    error: Option<&str>,
) -> Result<(), Error> {
    sqlx::query!(
        r#"
        UPDATE tensorzero.cleanup_run_steps
        SET status = $2, error = $3, finished_at = now()
        WHERE id = $1
        "#,
        step_id,
        status,
        error,
    )
    .execute(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to finish cleanup step `{step_id}`: {e}"),
        })
    })?;
    Ok(())
}

/// Lists recent cleanup runs, newest first.
pub async fn list_cleanup_runs(pool: &PgPool, limit: i64) -> Result<Vec<CleanupRunRow>, Error> {
    sqlx::query_as!(
        CleanupRunRow,
        r#"
        SELECT id, trigger, status, started_at, finished_at, error
        FROM tensorzero.cleanup_runs
        ORDER BY started_at DESC
        LIMIT $1
        "#,
        limit,
    )
    .fetch_all(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to list cleanup runs: {e}"),
        })
    })
}

/// Lists the steps of the given runs, oldest first.
pub async fn list_cleanup_run_steps(
    pool: &PgPool,
    run_ids: &[Uuid],
) -> Result<Vec<CleanupRunStepRow>, Error> {
    if run_ids.is_empty() {
        return Ok(vec![]);
    }
    sqlx::query_as!(
        CleanupRunStepRow,
        r#"
        SELECT id, run_id, rule_id, table_name, status, total_rows, rows_deleted,
               started_at, finished_at, error
        FROM tensorzero.cleanup_run_steps
        WHERE run_id = ANY($1)
        ORDER BY started_at
        "#,
        run_ids,
    )
    .fetch_all(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to list cleanup run steps: {e}"),
        })
    })
}

// --- Two-phase cleanup ---

/// Applies one cleanup rule to every target table, recording one
/// `cleanup_run_steps` row per table. Returns on the first failure (that step
/// is marked failed with the rows deleted so far preserved); the caller marks
/// the whole run failed.
pub async fn cleanup_rule(
    pool: &PgPool,
    rule: &CleanupRuleRow,
    run_id: Uuid,
    shutdown_token: &CancellationToken,
) -> Result<(), Error> {
    // Compute the age cutoff once so the ID search and every delete chunk use
    // the same bound.
    let cutoff = Utc::now() - chrono::Duration::days(i64::from(rule.older_than_days));
    for table in CleanupTargetTable::ALL {
        cleanup_table(pool, rule, run_id, table, cutoff, shutdown_token).await?;
    }
    Ok(())
}

async fn cleanup_table(
    pool: &PgPool,
    rule: &CleanupRuleRow,
    run_id: Uuid,
    table: CleanupTargetTable,
    cutoff: DateTime<Utc>,
    shutdown_token: &CancellationToken,
) -> Result<(), Error> {
    // Phase 1: collect the full set of ids to delete.
    // NOTE: the id set is materialized in memory (16 bytes per id, so ~16 MB
    // per 1M rows). Acceptable for v1; if rules regularly match far more than
    // that, switch to streaming/keyset pagination.
    let ids = select_cleanup_ids(
        pool,
        table,
        &rule.tag_key,
        rule.tag_value.as_deref(),
        cutoff,
    )
    .await?;
    let step_id = start_cleanup_step(pool, run_id, rule.id, table, ids.len() as i64).await?;

    // Phase 2: delete in small chunks, reporting progress after each one.
    let result = delete_ids_in_chunks(pool, table, &ids, cutoff, step_id, shutdown_token).await;
    match result {
        Ok(()) => finish_cleanup_step(pool, step_id, CLEANUP_STEP_STATUS_DONE, None).await?,
        Err(e) => {
            let message = e.to_string();
            if let Err(finish_error) =
                finish_cleanup_step(pool, step_id, CLEANUP_STEP_STATUS_FAILED, Some(&message)).await
            {
                tracing::error!(
                    "Failed to mark cleanup step `{step_id}` as failed: {finish_error}"
                );
            }
            return Err(e);
        }
    }
    Ok(())
}

async fn delete_ids_in_chunks(
    pool: &PgPool,
    table: CleanupTargetTable,
    ids: &[Uuid],
    cutoff: DateTime<Utc>,
    step_id: Uuid,
    shutdown_token: &CancellationToken,
) -> Result<(), Error> {
    let mut rows_deleted: i64 = 0;
    let mut chunks = ids.chunks(CLEANUP_DELETE_CHUNK_SIZE).peekable();
    while let Some(chunk) = chunks.next() {
        if shutdown_token.is_cancelled() {
            return Err(Error::new(ErrorDetails::InternalError {
                message: "Cleanup aborted: gateway is shutting down".to_string(),
            }));
        }
        // Already-deleted rows are naturally absent from the next run's search,
        // so an interrupted pass is idempotent.
        let deleted = delete_cleanup_chunk(pool, table, chunk, cutoff).await?;
        rows_deleted += deleted as i64;
        update_cleanup_step_progress(pool, step_id, rows_deleted).await?;
        if chunks.peek().is_some() {
            tokio::time::sleep(CLEANUP_CHUNK_DELAY).await;
        }
    }
    Ok(())
}

/// Searches a target table for the ids of rows matching the rule, bounded by
/// `cutoff` and the start of the current day.
async fn select_cleanup_ids(
    pool: &PgPool,
    table: CleanupTargetTable,
    tag_key: &str,
    tag_value: Option<&str>,
    cutoff: DateTime<Utc>,
) -> Result<Vec<Uuid>, Error> {
    let result = match table {
        CleanupTargetTable::ChatInferenceData => {
            sqlx::query_scalar!(
                r#"
                SELECT d.id
                FROM tensorzero.chat_inference_data d
                JOIN tensorzero.chat_inferences i ON i.id = d.id
                WHERE i.tags ? $1
                  AND ($2::TEXT IS NULL OR i.tags ->> $1 = $2)
                  AND i.protected_at IS NULL
                  AND d.created_at < $3
                  AND d.created_at < date_trunc('day', now())
                "#,
                tag_key,
                tag_value,
                cutoff,
            )
            .fetch_all(pool)
            .await
        }
        CleanupTargetTable::JsonInferenceData => {
            sqlx::query_scalar!(
                r#"
                SELECT d.id
                FROM tensorzero.json_inference_data d
                JOIN tensorzero.json_inferences i ON i.id = d.id
                WHERE i.tags ? $1
                  AND ($2::TEXT IS NULL OR i.tags ->> $1 = $2)
                  AND i.protected_at IS NULL
                  AND d.created_at < $3
                  AND d.created_at < date_trunc('day', now())
                "#,
                tag_key,
                tag_value,
                cutoff,
            )
            .fetch_all(pool)
            .await
        }
        CleanupTargetTable::ModelInferenceData => {
            sqlx::query_scalar!(
                r#"
                SELECT d.id AS "id!"
                FROM tensorzero.model_inference_data d
                JOIN tensorzero.model_inferences mi ON mi.id = d.id
                JOIN tensorzero.chat_inferences i ON i.id = mi.inference_id
                WHERE i.tags ? $1
                  AND ($2::TEXT IS NULL OR i.tags ->> $1 = $2)
                  AND i.protected_at IS NULL
                  AND d.created_at < $3
                  AND d.created_at < date_trunc('day', now())
                UNION
                SELECT d.id
                FROM tensorzero.model_inference_data d
                JOIN tensorzero.model_inferences mi ON mi.id = d.id
                JOIN tensorzero.json_inferences i ON i.id = mi.inference_id
                WHERE i.tags ? $1
                  AND ($2::TEXT IS NULL OR i.tags ->> $1 = $2)
                  AND i.protected_at IS NULL
                  AND d.created_at < $3
                  AND d.created_at < date_trunc('day', now())
                "#,
                tag_key,
                tag_value,
                cutoff,
            )
            .fetch_all(pool)
            .await
        }
        CleanupTargetTable::BatchModelInferenceData => {
            sqlx::query_scalar!(
                r#"
                SELECT d.inference_id
                FROM tensorzero.batch_model_inference_data d
                JOIN tensorzero.batch_model_inferences bmi ON bmi.inference_id = d.inference_id
                WHERE bmi.tags ? $1
                  AND ($2::TEXT IS NULL OR bmi.tags ->> $1 = $2)
                  AND d.created_at < $3
                  AND d.created_at < date_trunc('day', now())
                "#,
                tag_key,
                tag_value,
                cutoff,
            )
            .fetch_all(pool)
            .await
        }
        CleanupTargetTable::BatchRequestData => {
            sqlx::query_scalar!(
                r#"
                SELECT d.id
                FROM tensorzero.batch_request_data d
                JOIN tensorzero.batch_requests br ON br.id = d.id
                WHERE br.batch_id IN (
                    SELECT batch_id
                    FROM tensorzero.batch_model_inferences
                    WHERE tags ? $1
                      AND ($2::TEXT IS NULL OR tags ->> $1 = $2)
                )
                  AND d.created_at < $3
                  AND d.created_at < date_trunc('day', now())
                "#,
                tag_key,
                tag_value,
                cutoff,
            )
            .fetch_all(pool)
            .await
        }
    };
    result.map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!(
                "Failed to search `{}` for cleanup candidates: {e}",
                table.name()
            ),
        })
    })
}

/// Deletes one chunk of ids from a target table. The `created_at` bounds are
/// repeated here so partition pruning still applies and a row can never be
/// deleted from the current day's partition, even if it was inserted between
/// the search and the delete.
async fn delete_cleanup_chunk(
    pool: &PgPool,
    table: CleanupTargetTable,
    ids: &[Uuid],
    cutoff: DateTime<Utc>,
) -> Result<u64, Error> {
    let result = match table {
        CleanupTargetTable::ChatInferenceData => {
            sqlx::query!(
                r#"
                DELETE FROM tensorzero.chat_inference_data
                WHERE id = ANY($1)
                  AND created_at < $2
                  AND created_at < date_trunc('day', now())
                "#,
                ids,
                cutoff,
            )
            .execute(pool)
            .await
        }
        CleanupTargetTable::JsonInferenceData => {
            sqlx::query!(
                r#"
                DELETE FROM tensorzero.json_inference_data
                WHERE id = ANY($1)
                  AND created_at < $2
                  AND created_at < date_trunc('day', now())
                "#,
                ids,
                cutoff,
            )
            .execute(pool)
            .await
        }
        CleanupTargetTable::ModelInferenceData => {
            sqlx::query!(
                r#"
                DELETE FROM tensorzero.model_inference_data
                WHERE id = ANY($1)
                  AND created_at < $2
                  AND created_at < date_trunc('day', now())
                "#,
                ids,
                cutoff,
            )
            .execute(pool)
            .await
        }
        CleanupTargetTable::BatchModelInferenceData => {
            sqlx::query!(
                r#"
                DELETE FROM tensorzero.batch_model_inference_data
                WHERE inference_id = ANY($1)
                  AND created_at < $2
                  AND created_at < date_trunc('day', now())
                "#,
                ids,
                cutoff,
            )
            .execute(pool)
            .await
        }
        CleanupTargetTable::BatchRequestData => {
            sqlx::query!(
                r#"
                DELETE FROM tensorzero.batch_request_data
                WHERE id = ANY($1)
                  AND created_at < $2
                  AND created_at < date_trunc('day', now())
                "#,
                ids,
                cutoff,
            )
            .execute(pool)
            .await
        }
    };
    let result = result.map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!(
                "Failed to delete cleanup chunk from `{}`: {e}",
                table.name()
            ),
        })
    })?;
    Ok(result.rows_affected())
}

#[cfg(test)]
mod tests {
    use super::*;
    use googletest::prelude::*;

    #[gtest]
    fn target_table_names_match_daily_payload_tables() {
        let names: Vec<&'static str> = CleanupTargetTable::ALL
            .iter()
            .map(|table| table.name())
            .collect();
        expect_that!(
            names,
            eq(&vec![
                "chat_inference_data",
                "json_inference_data",
                "model_inference_data",
                "batch_model_inference_data",
                "batch_request_data",
            ])
        );
    }

    #[gtest]
    fn chunking_splits_into_delete_sized_pieces() {
        let ids: Vec<Uuid> = (0..2500).map(|_| Uuid::now_v7()).collect();
        let chunk_sizes: Vec<usize> = ids
            .chunks(CLEANUP_DELETE_CHUNK_SIZE)
            .map(<[Uuid]>::len)
            .collect();
        expect_that!(chunk_sizes, eq(&vec![1000, 1000, 500]));

        let empty: Vec<Uuid> = vec![];
        expect_that!(empty.chunks(CLEANUP_DELETE_CHUNK_SIZE).count(), eq(0));
    }
}

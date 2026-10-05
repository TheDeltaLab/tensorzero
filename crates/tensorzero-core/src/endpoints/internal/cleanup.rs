// Modified by Delta-AI under Apache 2.0
//! Internal endpoints for tag-based payload cleanup: cleanup rule CRUD,
//! manual run triggering, and run history/progress for the dashboard.

use std::collections::HashMap;

use axum::Json;
use axum::extract::{Path, Query, State};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use tracing::instrument;
use uuid::Uuid;

use crate::db::postgres::cleanup::{
    CleanupRuleRow, CleanupRunRow, CleanupRunStepRow, create_cleanup_rule, delete_cleanup_rule,
    list_cleanup_rules, list_cleanup_run_steps, list_cleanup_runs, update_cleanup_rule,
};
use crate::error::{Error, ErrorDetails};
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};

const DEFAULT_RUNS_LIMIT: i64 = 20;
const MAX_RUNS_LIMIT: i64 = 100;

/// A tag-based cleanup rule, as stored in `tensorzero.cleanup_rules`.
#[derive(ts_rs::TS, Debug, Serialize, Deserialize)]
#[ts(export, optional_fields)]
pub struct CleanupRule {
    pub id: Uuid,
    pub tag_key: String,
    /// When absent, the rule matches every row carrying `tag_key`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tag_value: Option<String>,
    pub older_than_days: i32,
    pub enabled: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl From<CleanupRuleRow> for CleanupRule {
    fn from(row: CleanupRuleRow) -> Self {
        Self {
            id: row.id,
            tag_key: row.tag_key,
            tag_value: row.tag_value,
            older_than_days: row.older_than_days,
            enabled: row.enabled,
            created_at: row.created_at,
            updated_at: row.updated_at,
        }
    }
}

/// Response for `GET /internal/cleanup/rules`.
#[derive(ts_rs::TS, Debug, Serialize, Deserialize)]
#[ts(export)]
pub struct ListCleanupRulesResponse {
    pub rules: Vec<CleanupRule>,
}

/// Request body for `POST /internal/cleanup/rules`.
#[derive(ts_rs::TS, Debug, Serialize, Deserialize)]
#[ts(export, optional_fields)]
pub struct CreateCleanupRuleRequest {
    pub tag_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tag_value: Option<String>,
    pub older_than_days: i32,
    /// Defaults to `true` when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
}

/// Request body for `PUT /internal/cleanup/rules/{rule_id}`.
/// Full-replace semantics: every field overwrites the stored rule
/// (`tag_value: null` clears the value filter).
#[derive(ts_rs::TS, Debug, Serialize, Deserialize)]
#[ts(export, optional_fields)]
pub struct UpdateCleanupRuleRequest {
    pub tag_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tag_value: Option<String>,
    pub older_than_days: i32,
    pub enabled: bool,
}

/// Response for `DELETE /internal/cleanup/rules/{rule_id}`.
#[derive(ts_rs::TS, Debug, Serialize, Deserialize)]
#[ts(export)]
pub struct DeleteCleanupRuleResponse {
    pub deleted: bool,
}

/// Response for `POST /internal/cleanup/run`.
#[derive(ts_rs::TS, Debug, Serialize, Deserialize)]
#[ts(export)]
pub struct TriggerCleanupRunResponse {
    /// Whether `[gateway.cleanup]` is currently enabled. When `false`, no
    /// worker is running, so the manual trigger has no effect.
    pub cleanup_enabled: bool,
}

/// Per-(run, rule, table) progress of a cleanup pass.
#[derive(ts_rs::TS, Debug, Serialize, Deserialize)]
#[ts(export, optional_fields)]
pub struct CleanupRunStep {
    pub id: Uuid,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rule_id: Option<Uuid>,
    pub table_name: String,
    /// `pending` / `running` / `done` / `failed`.
    pub status: String,
    /// Total rows selected for deletion (phase 1).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_rows: Option<i64>,
    /// Rows actually deleted so far (phase 2, updated after each chunk).
    pub rows_deleted: i64,
    pub started_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl From<CleanupRunStepRow> for CleanupRunStep {
    fn from(row: CleanupRunStepRow) -> Self {
        Self {
            id: row.id,
            rule_id: row.rule_id,
            table_name: row.table_name,
            status: row.status,
            total_rows: row.total_rows,
            rows_deleted: row.rows_deleted,
            started_at: row.started_at,
            finished_at: row.finished_at,
            error: row.error,
        }
    }
}

/// One cleanup pass with its per-table steps.
#[derive(ts_rs::TS, Debug, Serialize, Deserialize)]
#[ts(export, optional_fields)]
pub struct CleanupRun {
    pub id: Uuid,
    /// `schedule` / `manual`.
    pub trigger: String,
    /// `running` / `completed` / `failed`.
    pub status: String,
    pub started_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub steps: Vec<CleanupRunStep>,
}

/// Response for `GET /internal/cleanup/runs`.
#[derive(ts_rs::TS, Debug, Serialize, Deserialize)]
#[ts(export)]
pub struct ListCleanupRunsResponse {
    pub runs: Vec<CleanupRun>,
}

/// Query parameters for `GET /internal/cleanup/runs`.
#[derive(Debug, Deserialize)]
pub struct ListCleanupRunsParams {
    /// Number of recent runs to return (default 20, capped at 100).
    pub limit: Option<i64>,
}

fn postgres_required(app_state: &AppStateData) -> Result<&PgPool, Error> {
    let Some(pool) = app_state.postgres_connection_info.get_pool() else {
        return Err(Error::new(ErrorDetails::PostgresConnection {
            message: "Postgres is required for tag-based cleanup".to_string(),
        }));
    };
    Ok(pool)
}

fn validate_rule_fields(tag_key: &str, older_than_days: i32) -> Result<(), Error> {
    if tag_key.is_empty() {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: "`tag_key` must be non-empty".to_string(),
        }));
    }
    if older_than_days < 1 {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: "`older_than_days` must be at least 1".to_string(),
        }));
    }
    Ok(())
}

fn clamp_runs_limit(limit: Option<i64>) -> i64 {
    limit.unwrap_or(DEFAULT_RUNS_LIMIT).clamp(1, MAX_RUNS_LIMIT)
}

/// Handler for `GET /internal/cleanup/rules`
#[instrument(name = "list_cleanup_rules", skip_all)]
pub async fn list_cleanup_rules_handler(
    State(app_state): AppState,
) -> Result<Json<ListCleanupRulesResponse>, Error> {
    let pool = postgres_required(&app_state)?;
    let rules = list_cleanup_rules(pool)
        .await?
        .into_iter()
        .map(CleanupRule::from)
        .collect();
    Ok(Json(ListCleanupRulesResponse { rules }))
}

/// Handler for `POST /internal/cleanup/rules`
#[instrument(name = "create_cleanup_rule", skip_all)]
pub async fn create_cleanup_rule_handler(
    State(app_state): AppState,
    StructuredJson(request): StructuredJson<CreateCleanupRuleRequest>,
) -> Result<Json<CleanupRule>, Error> {
    let pool = postgres_required(&app_state)?;
    validate_rule_fields(&request.tag_key, request.older_than_days)?;

    let rule = create_cleanup_rule(
        pool,
        &request.tag_key,
        request.tag_value.as_deref(),
        request.older_than_days,
        request.enabled.unwrap_or(true),
    )
    .await?;
    Ok(Json(CleanupRule::from(rule)))
}

/// Handler for `PUT /internal/cleanup/rules/{rule_id}`
#[instrument(name = "update_cleanup_rule", skip_all)]
pub async fn update_cleanup_rule_handler(
    State(app_state): AppState,
    Path(rule_id): Path<Uuid>,
    StructuredJson(request): StructuredJson<UpdateCleanupRuleRequest>,
) -> Result<Json<CleanupRule>, Error> {
    let pool = postgres_required(&app_state)?;
    validate_rule_fields(&request.tag_key, request.older_than_days)?;

    let Some(rule) = update_cleanup_rule(
        pool,
        rule_id,
        &request.tag_key,
        request.tag_value.as_deref(),
        request.older_than_days,
        request.enabled,
    )
    .await?
    else {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!("Cleanup rule `{rule_id}` was not found"),
        }));
    };
    Ok(Json(CleanupRule::from(rule)))
}

/// Handler for `DELETE /internal/cleanup/rules/{rule_id}`
#[instrument(name = "delete_cleanup_rule", skip_all)]
pub async fn delete_cleanup_rule_handler(
    State(app_state): AppState,
    Path(rule_id): Path<Uuid>,
) -> Result<Json<DeleteCleanupRuleResponse>, Error> {
    let pool = postgres_required(&app_state)?;
    if !delete_cleanup_rule(pool, rule_id).await? {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!("Cleanup rule `{rule_id}` was not found"),
        }));
    }
    Ok(Json(DeleteCleanupRuleResponse { deleted: true }))
}

/// Handler for `POST /internal/cleanup/run`
///
/// Notifies the cleanup worker to start a pass immediately and returns without
/// waiting for it; progress is polled via `GET /internal/cleanup/runs`.
#[instrument(name = "trigger_cleanup_run", skip_all)]
pub async fn trigger_cleanup_run_handler(
    State(app_state): AppState,
) -> Result<Json<TriggerCleanupRunResponse>, Error> {
    postgres_required(&app_state)?;
    app_state.cleanup_notify.notify_one();
    let cleanup_enabled = app_state.config.gateway.cleanup.enabled;
    Ok(Json(TriggerCleanupRunResponse { cleanup_enabled }))
}

/// Handler for `GET /internal/cleanup/runs`
#[instrument(name = "list_cleanup_runs", skip_all)]
pub async fn list_cleanup_runs_handler(
    State(app_state): AppState,
    Query(params): Query<ListCleanupRunsParams>,
) -> Result<Json<ListCleanupRunsResponse>, Error> {
    let pool = postgres_required(&app_state)?;
    let limit = clamp_runs_limit(params.limit);

    let runs = list_cleanup_runs(pool, limit).await?;
    let run_ids: Vec<Uuid> = runs.iter().map(|run| run.id).collect();
    let steps = list_cleanup_run_steps(pool, &run_ids).await?;

    let mut steps_by_run: HashMap<Uuid, Vec<CleanupRunStepRow>> = HashMap::new();
    for step in steps {
        steps_by_run.entry(step.run_id).or_default().push(step);
    }

    let runs = runs
        .into_iter()
        .map(|run| {
            let steps = steps_by_run
                .remove(&run.id)
                .unwrap_or_default()
                .into_iter()
                .map(CleanupRunStep::from)
                .collect();
            cleanup_run_from_row(run, steps)
        })
        .collect();
    Ok(Json(ListCleanupRunsResponse { runs }))
}

fn cleanup_run_from_row(row: CleanupRunRow, steps: Vec<CleanupRunStep>) -> CleanupRun {
    CleanupRun {
        id: row.id,
        trigger: row.trigger,
        status: row.status,
        started_at: row.started_at,
        finished_at: row.finished_at,
        error: row.error,
        steps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use googletest::prelude::*;
    use googletest_matchers::matches_json_literal;

    #[gtest]
    fn validate_rule_fields_rejects_empty_tag_key() {
        expect_that!(validate_rule_fields("", 30), err(anything()));
        expect_that!(validate_rule_fields("env", 30), ok(eq(&())));
    }

    #[gtest]
    fn validate_rule_fields_rejects_zero_older_than_days() {
        expect_that!(validate_rule_fields("env", 0), err(anything()));
        expect_that!(validate_rule_fields("env", -5), err(anything()));
        expect_that!(validate_rule_fields("env", 1), ok(eq(&())));
    }

    #[gtest]
    fn runs_limit_defaults_and_clamps() {
        expect_that!(clamp_runs_limit(None), eq(20));
        expect_that!(clamp_runs_limit(Some(50)), eq(50));
        expect_that!(clamp_runs_limit(Some(0)), eq(1));
        expect_that!(clamp_runs_limit(Some(-3)), eq(1));
        expect_that!(clamp_runs_limit(Some(10_000)), eq(100));
    }

    #[gtest]
    fn cleanup_run_serializes_with_optional_fields_omitted() {
        let started_at = DateTime::parse_from_rfc3339("2026-10-04T10:00:00Z")
            .expect("valid timestamp")
            .with_timezone(&Utc);
        let run = CleanupRun {
            id: Uuid::parse_str("0190f9c4-8e3a-7b3d-9c1e-2f4a5b6c7d8e").expect("valid UUID"),
            trigger: "schedule".to_string(),
            status: "running".to_string(),
            started_at,
            finished_at: None,
            error: None,
            steps: vec![CleanupRunStep {
                id: Uuid::parse_str("0190f9c4-8e3a-7b3d-9c1e-2f4a5b6c7d8f").expect("valid UUID"),
                rule_id: None,
                table_name: "chat_inference_data".to_string(),
                status: "running".to_string(),
                total_rows: Some(2500),
                rows_deleted: 1000,
                started_at,
                finished_at: None,
                error: None,
            }],
        };
        let value = serde_json::to_value(&run).expect("should serialize");
        expect_that!(
            value,
            matches_json_literal!({
                "id": "0190f9c4-8e3a-7b3d-9c1e-2f4a5b6c7d8e",
                "trigger": "schedule",
                "status": "running",
                "started_at": "2026-10-04T10:00:00Z",
                "steps": [{
                    "id": "0190f9c4-8e3a-7b3d-9c1e-2f4a5b6c7d8f",
                    "table_name": "chat_inference_data",
                    "status": "running",
                    "total_rows": 2500,
                    "rows_deleted": 1000,
                    "started_at": "2026-10-04T10:00:00Z"
                }]
            })
        );
        expect_that!(value.get("finished_at"), none());
        expect_that!(value.get("error"), none());
        expect_that!(value["steps"][0].get("rule_id"), none());
        expect_that!(value["steps"][0].get("finished_at"), none());
    }
}

// Modified by Delta-AI under Apache 2.0
//! `GET /internal/async_tasks`: list async inference tasks from the durable
//! queue backing `[gateway.async_inference]`, for the dashboard's async tasks
//! page. Row-level detail stays on the public `GET /v1/async_tasks/{task_id}`.

use axum::Json;
use axum::extract::{Query, State};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use sqlx::{PgPool, QueryBuilder};
use tracing::instrument;
use ts_rs::TS;
use uuid::Uuid;

use crate::endpoints::openai_compatible::async_inference_types::AsyncInferenceApiKind;
use crate::error::{Error, ErrorDetails};
use crate::utils::gateway::AppState;

const DEFAULT_LIMIT: i64 = 50;
const MAX_LIMIT: i64 = 200;

/// List status of an async inference task, rolling the durable queue's
/// `pending`/`sleeping` states up into `queued`. Matches the status tags of
/// the public `GET /v1/async_tasks/{task_id}` response.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, Serialize, TS)]
#[ts(export)]
#[serde(rename_all = "snake_case")]
pub enum AsyncTaskStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl AsyncTaskStatus {
    /// Durable task-table `state` values covered by this status.
    fn states(self) -> Vec<&'static str> {
        match self {
            Self::Queued => vec!["pending", "sleeping"],
            Self::Running => vec!["running"],
            Self::Completed => vec!["completed"],
            Self::Failed => vec!["failed"],
            Self::Cancelled => vec!["cancelled"],
        }
    }

    fn from_state(state: &str) -> Result<Self, Error> {
        match state {
            "pending" | "sleeping" => Ok(Self::Queued),
            "running" => Ok(Self::Running),
            "completed" => Ok(Self::Completed),
            "failed" => Ok(Self::Failed),
            "cancelled" => Ok(Self::Cancelled),
            other => Err(Error::new(ErrorDetails::InternalError {
                message: format!("Unknown async inference task state `{other}`"),
            })),
        }
    }
}

/// Query parameters for `GET /internal/async_tasks`.
#[derive(Debug, Deserialize)]
pub struct ListAsyncTasksParams {
    /// Page size (default 50, capped at 200).
    pub limit: Option<i64>,
    /// Offset into the result set, ordered by `enqueue_at` DESC.
    pub offset: Option<i64>,
    /// Only list tasks in this status.
    pub status: Option<AsyncTaskStatus>,
}

/// One row of the `GET /internal/async_tasks` response.
#[derive(Debug, Serialize, TS)]
#[ts(export, optional_fields)]
pub struct AsyncTaskSummary {
    pub task_id: Uuid,
    pub status: AsyncTaskStatus,
    /// Which API shape the task executes (`chat` / `responses` / `messages`),
    /// from the stored task params.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub api_kind: Option<AsyncInferenceApiKind>,
    /// Model from the stored request body, exactly as submitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub model: Option<String>,
    pub enqueue_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub started_at: Option<DateTime<Utc>>,
    /// Elapsed time for running tasks, total execution time for terminal
    /// tasks. `None` while the task has never started.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub duration_ms: Option<u64>,
    /// Failure message of the latest run, for failed/cancelled tasks.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub error: Option<String>,
    /// Inference written by a completed task, from the stored final response
    /// (`id` for chat/responses, `msg_`-prefixed id for messages).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub inference_id: Option<String>,
    pub attempts: i32,
}

/// Response of `GET /internal/async_tasks`.
#[derive(Debug, Serialize, TS)]
#[ts(export)]
pub struct ListAsyncTasksResponse {
    pub tasks: Vec<AsyncTaskSummary>,
}

/// Raw row read from the durable queue tables (`t_<queue>` joined with the
/// latest run of `r_<queue>`).
#[derive(Debug, sqlx::FromRow)]
struct AsyncTaskRow {
    task_id: Uuid,
    state: String,
    params: JsonValue,
    enqueue_at: DateTime<Utc>,
    first_started_at: Option<DateTime<Utc>>,
    cancelled_at: Option<DateTime<Utc>>,
    completed_payload: Option<JsonValue>,
    attempts: i32,
    finished_at: Option<DateTime<Utc>>,
    failure_message: Option<String>,
}

/// Handler for `GET /internal/async_tasks`.
#[instrument(name = "list_async_tasks", skip_all)]
pub async fn list_async_tasks_handler(
    State(app_state): AppState,
    Query(params): Query<ListAsyncTasksParams>,
) -> Result<Json<ListAsyncTasksResponse>, Error> {
    // Async inference needs Postgres for the durable queue; without it there
    // are no tasks to list.
    let Some(pool) = app_state.postgres_connection_info.get_pool() else {
        return Ok(Json(ListAsyncTasksResponse { tasks: vec![] }));
    };
    let queue_name = app_state.config.gateway.async_inference.queue_name.clone();
    let tasks = list_async_tasks(pool, &queue_name, &params).await?;
    Ok(Json(ListAsyncTasksResponse { tasks }))
}

async fn list_async_tasks(
    pool: &PgPool,
    queue_name: &str,
    params: &ListAsyncTasksParams,
) -> Result<Vec<AsyncTaskSummary>, Error> {
    let limit = params.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT);
    let offset = params.offset.unwrap_or(0).max(0);

    let mut query = build_list_query(queue_name, params.status, limit, offset);
    let rows: Vec<AsyncTaskRow> = query
        .build_query_as::<AsyncTaskRow>()
        .fetch_all(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to list async inference tasks: {e}"),
            })
        })?;

    let now = Utc::now();
    rows.into_iter().map(|row| summarize(row, now)).collect()
}

/// Build the list query. Table names are derived from the operator-provided
/// queue name (a trusted config value, already validated when the durable
/// queue was created); all user-provided values are bound.
fn build_list_query(
    queue_name: &str,
    status: Option<AsyncTaskStatus>,
    limit: i64,
    offset: i64,
) -> QueryBuilder<sqlx::Postgres> {
    let mut query = QueryBuilder::new(
        "SELECT t.task_id, t.state, t.params, t.enqueue_at, t.first_started_at, \
         t.cancelled_at, t.completed_payload, t.attempts, r.finished_at, r.failure_message \
         FROM durable.t_",
    );
    query.push(queue_name);
    query.push(
        " t LEFT JOIN LATERAL (\
         SELECT COALESCE(x.failed_at, x.completed_at) AS finished_at, \
         COALESCE(x.failure_reason->>'message', x.failure_reason::text) AS failure_message \
         FROM durable.r_",
    );
    query.push(queue_name);
    query.push(
        " x WHERE x.task_id = t.task_id ORDER BY x.attempt DESC LIMIT 1\
         ) r ON TRUE",
    );
    if let Some(status) = status {
        query.push(" WHERE t.state = ANY(");
        query.push_bind(status.states());
        query.push(")");
    }
    query.push(" ORDER BY t.enqueue_at DESC LIMIT ");
    query.push_bind(limit);
    query.push(" OFFSET ");
    query.push_bind(offset);
    query
}

/// Extract the inference id from a completed task's final response payload:
/// a bare UUID for chat/responses, `msg_`-prefixed for messages. Defensive by
/// construction — a missing or malformed payload yields `None`, never an
/// error, so one bad row cannot fail the whole list.
fn extract_inference_id(payload: Option<&JsonValue>) -> Option<String> {
    let raw = payload?.get("id")?.as_str()?;
    let id = raw.strip_prefix("msg_").unwrap_or(raw);
    let uuid = Uuid::parse_str(id).ok()?;
    Some(uuid.to_string())
}

/// Map a raw durable-queue row to the API shape, extracting `api_kind` and
/// `model` from the stored task params.
fn summarize(row: AsyncTaskRow, now: DateTime<Utc>) -> Result<AsyncTaskSummary, Error> {
    let status = AsyncTaskStatus::from_state(&row.state)?;
    let api_kind = row
        .params
        .get("api_kind")
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(|e| {
            Error::new(ErrorDetails::InternalError {
                message: format!(
                    "Malformed `api_kind` in async inference task `{}` params: {e}",
                    row.task_id
                ),
            })
        })?;
    let model = row
        .params
        .pointer("/request/model")
        .and_then(JsonValue::as_str)
        .map(str::to_string);

    let duration_ms = match status {
        AsyncTaskStatus::Running => row
            .first_started_at
            .map(|started| now.signed_duration_since(started).num_milliseconds().max(0) as u64),
        AsyncTaskStatus::Completed | AsyncTaskStatus::Failed | AsyncTaskStatus::Cancelled => {
            let finished_at = row.finished_at.or(row.cancelled_at);
            match (row.first_started_at, finished_at) {
                (Some(started), Some(finished)) => Some(
                    finished
                        .signed_duration_since(started)
                        .num_milliseconds()
                        .max(0) as u64,
                ),
                _ => None,
            }
        }
        AsyncTaskStatus::Queued => None,
    };

    Ok(AsyncTaskSummary {
        task_id: row.task_id,
        status,
        api_kind,
        model,
        enqueue_at: row.enqueue_at,
        started_at: row.first_started_at,
        duration_ms,
        error: row.failure_message,
        inference_id: extract_inference_id(row.completed_payload.as_ref()),
        attempts: row.attempts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use googletest::prelude::*;
    use googletest_matchers::{matches_json, matches_json_literal};
    use serde_json::json;

    fn test_task_id() -> Uuid {
        Uuid::parse_str("0190f9c4-8e3a-7b3d-9c1e-2f4a5b6c7d8e").expect("valid UUID")
    }

    fn test_now() -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2026-09-04T10:00:00Z")
            .expect("valid timestamp")
            .with_timezone(&Utc)
    }

    fn row(state: &str, params: JsonValue) -> AsyncTaskRow {
        AsyncTaskRow {
            task_id: test_task_id(),
            state: state.to_string(),
            params,
            enqueue_at: test_now(),
            first_started_at: None,
            cancelled_at: None,
            completed_payload: None,
            attempts: 1,
            finished_at: None,
            failure_message: None,
        }
    }

    #[gtest]
    fn list_query_uses_configured_queue_tables() {
        let query = build_list_query("async_inference", None, 50, 0);
        let sql = query.sql();
        expect_that!(sql, contains_substring("FROM durable.t_async_inference t"));
        expect_that!(sql, contains_substring("FROM durable.r_async_inference x"));
        expect_that!(sql, contains_substring("ORDER BY t.enqueue_at DESC"));
        expect_that!(sql, not(contains_substring("WHERE t.state = ANY(")));
    }

    #[gtest]
    fn list_query_filters_by_mapped_states() {
        let query = build_list_query("custom_queue", Some(AsyncTaskStatus::Queued), 50, 100);
        let sql = query.sql();
        expect_that!(sql, contains_substring("durable.t_custom_queue"));
        expect_that!(sql, contains_substring("WHERE t.state = ANY("));
        expect_that!(
            AsyncTaskStatus::Queued.states(),
            eq(&vec!["pending", "sleeping"])
        );
        expect_that!(AsyncTaskStatus::Running.states(), eq(&vec!["running"]));
        expect_that!(AsyncTaskStatus::Completed.states(), eq(&vec!["completed"]));
        expect_that!(AsyncTaskStatus::Failed.states(), eq(&vec!["failed"]));
        expect_that!(AsyncTaskStatus::Cancelled.states(), eq(&vec!["cancelled"]));
    }

    #[gtest]
    fn from_state_rolls_up_pending_and_sleeping() {
        expect_that!(
            AsyncTaskStatus::from_state("pending").expect("known state"),
            eq(AsyncTaskStatus::Queued)
        );
        expect_that!(
            AsyncTaskStatus::from_state("sleeping").expect("known state"),
            eq(AsyncTaskStatus::Queued)
        );
        expect_that!(AsyncTaskStatus::from_state("bogus"), err(anything()));
    }

    #[gtest]
    fn summarize_extracts_api_kind_and_model_from_params() {
        let params = json!({
            "api_kind": "messages",
            "request": {"model": "anthropic::claude-sonnet-4-5", "messages": []},
            "headers": {},
        });
        let summary = summarize(row("pending", params), test_now()).expect("should summarize");
        expect_that!(summary.status, eq(AsyncTaskStatus::Queued));
        expect_that!(summary.api_kind, some(eq(AsyncInferenceApiKind::Messages)));
        expect_that!(
            summary.model.as_deref(),
            some(eq("anthropic::claude-sonnet-4-5"))
        );
        expect_that!(summary.duration_ms, none());
        expect_that!(summary.error, none());
    }

    #[gtest]
    fn summarize_running_task_reports_elapsed_time() {
        let mut running = row("running", json!({"api_kind": "chat", "request": {}}));
        running.first_started_at = Some(test_now() - chrono::Duration::milliseconds(1500));
        let summary = summarize(running, test_now()).expect("should summarize");
        expect_that!(summary.status, eq(AsyncTaskStatus::Running));
        expect_that!(summary.duration_ms, some(eq(1500)));
        expect_that!(
            summary.started_at,
            some(eq(test_now() - chrono::Duration::milliseconds(1500)))
        );
    }

    #[gtest]
    fn summarize_failed_task_reports_duration_and_error() {
        let started = test_now() - chrono::Duration::seconds(4);
        let finished = test_now() - chrono::Duration::seconds(1);
        let mut failed = row("failed", json!({"api_kind": "chat", "request": {}}));
        failed.first_started_at = Some(started);
        failed.finished_at = Some(finished);
        failed.failure_message = Some("model exploded".to_string());
        let summary = summarize(failed, test_now()).expect("should summarize");
        expect_that!(summary.status, eq(AsyncTaskStatus::Failed));
        expect_that!(summary.duration_ms, some(eq(3000)));
        expect_that!(summary.error.as_deref(), some(eq("model exploded")));
    }

    #[gtest]
    fn summarize_cancelled_task_falls_back_to_cancelled_at() {
        let started = test_now() - chrono::Duration::seconds(2);
        let mut cancelled = row("cancelled", json!({"api_kind": "chat", "request": {}}));
        cancelled.first_started_at = Some(started);
        cancelled.cancelled_at = Some(test_now());
        let summary = summarize(cancelled, test_now()).expect("should summarize");
        expect_that!(summary.status, eq(AsyncTaskStatus::Cancelled));
        expect_that!(summary.duration_ms, some(eq(2000)));
    }

    #[gtest]
    fn summarize_completed_chat_task_extracts_bare_inference_id() {
        let mut completed = row("completed", json!({"api_kind": "chat", "request": {}}));
        completed.completed_payload = Some(
            json!({"id": "018e5f1c-2b3a-7c4d-8e5f-6a7b8c9d0e1f", "object": "chat.completion"}),
        );
        let summary = summarize(completed, test_now()).expect("should summarize");
        expect_that!(
            summary.inference_id.as_deref(),
            some(eq("018e5f1c-2b3a-7c4d-8e5f-6a7b8c9d0e1f"))
        );
    }

    #[gtest]
    fn summarize_completed_messages_task_strips_msg_prefix() {
        let mut completed = row("completed", json!({"api_kind": "messages", "request": {}}));
        completed.completed_payload =
            Some(json!({"id": "msg_018e5f1c-2b3a-7c4d-8e5f-6a7b8c9d0e1f", "type": "message"}));
        let summary = summarize(completed, test_now()).expect("should summarize");
        expect_that!(
            summary.inference_id.as_deref(),
            some(eq("018e5f1c-2b3a-7c4d-8e5f-6a7b8c9d0e1f"))
        );
    }

    #[gtest]
    fn summarize_treats_missing_or_malformed_payload_as_no_inference() {
        // Queued/running/failed tasks have no stored payload.
        let queued = row("pending", json!({"api_kind": "chat", "request": {}}));
        expect_that!(
            summarize(queued, test_now())
                .expect("should summarize")
                .inference_id,
            none()
        );

        for payload in [
            json!(null),
            json!({}),
            json!({"id": 42}),
            json!({"id": "not-a-uuid"}),
            json!({"id": "msg_not-a-uuid"}),
        ] {
            expect_that!(
                extract_inference_id(Some(&payload)),
                none(),
                "payload {payload} should yield no inference id"
            );
        }
        expect_that!(extract_inference_id(None), none());
    }

    #[gtest]
    fn summarize_rejects_malformed_api_kind() {
        let bad = row("pending", json!({"api_kind": 42, "request": {}}));
        expect_that!(summarize(bad, test_now()), err(anything()));
    }

    #[gtest]
    fn summary_serializes_with_optional_fields_omitted() {
        let summary = AsyncTaskSummary {
            task_id: test_task_id(),
            status: AsyncTaskStatus::Queued,
            api_kind: Some(AsyncInferenceApiKind::Chat),
            model: Some("openai::gpt-5".to_string()),
            enqueue_at: test_now(),
            started_at: None,
            duration_ms: None,
            error: None,
            inference_id: None,
            attempts: 0,
        };
        let value = serde_json::to_value(&summary).expect("should serialize");
        expect_that!(
            value,
            matches_json!({
                "task_id": eq("0190f9c4-8e3a-7b3d-9c1e-2f4a5b6c7d8e"),
                "status": eq("queued"),
                "api_kind": eq("chat"),
                "model": eq("openai::gpt-5"),
                "enqueue_at": eq("2026-09-04T10:00:00Z"),
                "attempts": eq(0),
            })
        );
        expect_that!(value.get("started_at"), none());
        expect_that!(value.get("duration_ms"), none());
        expect_that!(value.get("error"), none());
        expect_that!(value.get("inference_id"), none());
    }

    #[gtest]
    fn response_serializes_task_list() {
        let response = ListAsyncTasksResponse {
            tasks: vec![AsyncTaskSummary {
                task_id: test_task_id(),
                status: AsyncTaskStatus::Failed,
                api_kind: None,
                model: None,
                enqueue_at: test_now(),
                started_at: None,
                duration_ms: None,
                error: Some("boom".to_string()),
                inference_id: None,
                attempts: 3,
            }],
        };
        let value = serde_json::to_value(&response).expect("should serialize");
        expect_that!(
            value,
            matches_json_literal!({
                "tasks": [{
                    "task_id": "0190f9c4-8e3a-7b3d-9c1e-2f4a5b6c7d8e",
                    "status": "failed",
                    "enqueue_at": "2026-09-04T10:00:00Z",
                    "error": "boom",
                    "attempts": 3
                }]
            })
        );
    }
}

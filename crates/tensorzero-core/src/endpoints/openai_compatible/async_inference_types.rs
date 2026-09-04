// Modified by Delta-AI under Apache 2.0
//! Types for the async inference API (`POST .../async` submit endpoints,
//! `GET /v1/async_tasks/{task_id}` status, `GET /v1/async_tasks/{task_id}/stream`).

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tensorzero_derive::TensorZeroDeserialize;
use ts_rs::TS;
use uuid::Uuid;

/// Which OpenAI/Anthropic-compatible API shape an async inference task
/// executes and returns.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize, TS)]
#[ts(export)]
#[serde(rename_all = "snake_case")]
pub enum AsyncInferenceApiKind {
    /// OpenAI chat completions (`POST /v1/chat/completions/async`)
    Chat,
    /// OpenAI responses (`POST /v1/responses/async`)
    Responses,
    /// Anthropic messages (`POST /v1/messages/async`)
    Messages,
}

/// Durable task parameters for an async inference, stored as the task's
/// `params` column in the durable queue.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AsyncInferenceTaskParams {
    pub api_kind: AsyncInferenceApiKind,
    /// The original request body, exactly as received by the submit endpoint.
    /// The `stream` field is ignored; execution is always streaming internally.
    pub request: Value,
    /// Synapse/TensorZero compatibility headers captured at submit time
    /// (`x-synapse-*`, `x-tensorzero-*`, `x-request-id`), used to rebuild the
    /// request context in the worker. Credentials (`authorization`, API keys)
    /// are never captured.
    pub headers: BTreeMap<String, String>,
    /// Public ID of the API key the submit request was authenticated with,
    /// stamped onto the inference's `tensorzero::api_key_public_id` tag by the
    /// worker so async inferences stay visible in API-key-filtered views.
    /// Only the public ID is stored, never the secret key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_public_id: Option<String>,
}

/// Response of the async submit endpoints (HTTP 202).
#[derive(Clone, Debug, Deserialize, Serialize, TS)]
#[ts(export)]
pub struct AsyncInferenceLaunchResponse {
    pub task_id: Uuid,
}

/// `GET /v1/async_tasks/{task_id}` response, internally tagged on `status`.
///
/// Serializes as `{"task_id": ..., "status": "queued", ...}` etc.
#[derive(Clone, Debug, Serialize, TensorZeroDeserialize, TS)]
#[ts(export, optional_fields)]
#[serde(tag = "status")]
#[serde(rename_all = "snake_case")]
pub enum AsyncTaskStatusResponse {
    Queued {
        task_id: Uuid,
        /// Number of claimable tasks ahead of this one (`None` if unknown).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        queue_position: Option<i64>,
    },
    Running {
        task_id: Uuid,
        /// RFC 3339 timestamp of when execution first started.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        started_at: Option<DateTime<Utc>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        elapsed_ms: Option<u64>,
    },
    Completed {
        task_id: Uuid,
        /// The final response, in the same shape as the synchronous API the
        /// task was submitted to.
        response: Value,
    },
    Failed {
        task_id: Uuid,
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        error: Option<Value>,
    },
    Cancelled {
        task_id: Uuid,
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        error: Option<Value>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use googletest::prelude::*;
    use googletest_matchers::matches_json;
    use serde_json::json;

    fn test_task_id() -> Uuid {
        Uuid::parse_str("0190f9c4-8e3a-7b3d-9c1e-2f4a5b6c7d8e").expect("valid UUID")
    }

    #[gtest]
    fn queued_status_includes_queue_position() {
        let response = AsyncTaskStatusResponse::Queued {
            task_id: test_task_id(),
            queue_position: Some(3),
        };
        let value = serde_json::to_value(&response).expect("should serialize");
        expect_that!(
            value,
            matches_json!({
                "status": eq("queued"),
                "task_id": eq("0190f9c4-8e3a-7b3d-9c1e-2f4a5b6c7d8e"),
                "queue_position": eq(3),
            })
        );
    }

    #[gtest]
    fn queued_status_omits_unknown_queue_position() {
        let response = AsyncTaskStatusResponse::Queued {
            task_id: test_task_id(),
            queue_position: None,
        };
        let value = serde_json::to_value(&response).expect("should serialize");
        expect_that!(
            value,
            matches_json!({
                "status": eq("queued"),
                "task_id": eq("0190f9c4-8e3a-7b3d-9c1e-2f4a5b6c7d8e"),
            })
        );
    }

    #[gtest]
    fn running_status_includes_timing() {
        let response = AsyncTaskStatusResponse::Running {
            task_id: test_task_id(),
            started_at: DateTime::parse_from_rfc3339("2026-09-03T10:00:00Z")
                .map(|t| t.with_timezone(&Utc))
                .ok(),
            elapsed_ms: Some(1500),
        };
        let value = serde_json::to_value(&response).expect("should serialize");
        expect_that!(
            value,
            matches_json!({
                "status": eq("running"),
                "task_id": eq("0190f9c4-8e3a-7b3d-9c1e-2f4a5b6c7d8e"),
                "started_at": eq("2026-09-03T10:00:00Z"),
                "elapsed_ms": eq(1500),
            })
        );
    }

    #[gtest]
    fn completed_status_carries_response() {
        let response = AsyncTaskStatusResponse::Completed {
            task_id: test_task_id(),
            response: json!({"id": "chatcmpl-123"}),
        };
        let value = serde_json::to_value(&response).expect("should serialize");
        expect_that!(
            value,
            matches_json!({
                "status": eq("completed"),
                "task_id": eq("0190f9c4-8e3a-7b3d-9c1e-2f4a5b6c7d8e"),
                "response": matches_json!({"id": eq("chatcmpl-123")}),
            })
        );
    }

    #[gtest]
    fn failed_status_omits_missing_error() {
        let response = AsyncTaskStatusResponse::Failed {
            task_id: test_task_id(),
            error: None,
        };
        let value = serde_json::to_value(&response).expect("should serialize");
        expect_that!(
            value,
            matches_json!({
                "status": eq("failed"),
                "task_id": eq("0190f9c4-8e3a-7b3d-9c1e-2f4a5b6c7d8e"),
            })
        );
    }

    #[gtest]
    fn cancelled_status_serializes_snake_case() {
        let response = AsyncTaskStatusResponse::Cancelled {
            task_id: test_task_id(),
            error: Some(json!({"message": "boom"})),
        };
        let value = serde_json::to_value(&response).expect("should serialize");
        expect_that!(
            value,
            matches_json!({
                "status": eq("cancelled"),
                "task_id": eq("0190f9c4-8e3a-7b3d-9c1e-2f4a5b6c7d8e"),
                "error": matches_json!({"message": eq("boom")}),
            })
        );
    }

    #[gtest]
    fn task_params_roundtrip_preserves_api_kind_and_headers() {
        let params = AsyncInferenceTaskParams {
            api_kind: AsyncInferenceApiKind::Chat,
            request: json!({"model": "openai::gpt-5", "messages": []}),
            headers: BTreeMap::from([("x-request-id".to_string(), "req-1".to_string())]),
            api_key_public_id: Some("abcdefghijkl".to_string()),
        };
        let value = serde_json::to_value(&params).expect("should serialize");
        expect_that!(&value["api_kind"], eq(&json!("chat")));
        expect_that!(&value["api_key_public_id"], eq(&json!("abcdefghijkl")));

        let parsed: AsyncInferenceTaskParams =
            serde_json::from_value(value).expect("should deserialize");
        expect_that!(parsed.api_kind, eq(AsyncInferenceApiKind::Chat));
        expect_that!(
            parsed.headers.get("x-request-id").map(String::as_str),
            some(eq("req-1"))
        );
        expect_that!(
            parsed.api_key_public_id.as_deref(),
            some(eq("abcdefghijkl"))
        );
    }

    #[gtest]
    fn task_params_omit_and_default_missing_api_key_public_id() {
        let params = AsyncInferenceTaskParams {
            api_kind: AsyncInferenceApiKind::Chat,
            request: json!({"model": "openai::gpt-5", "messages": []}),
            headers: BTreeMap::new(),
            api_key_public_id: None,
        };
        let value = serde_json::to_value(&params).expect("should serialize");
        expect_that!(value.get("api_key_public_id"), none());

        // Tasks enqueued before the field existed carry no `api_key_public_id`
        // and must still deserialize.
        let parsed: AsyncInferenceTaskParams =
            serde_json::from_value(value).expect("should deserialize");
        expect_that!(parsed.api_key_public_id, none());
    }

    #[gtest]
    fn api_kind_serializes_snake_case() {
        for (kind, expected) in [
            (AsyncInferenceApiKind::Chat, "chat"),
            (AsyncInferenceApiKind::Responses, "responses"),
            (AsyncInferenceApiKind::Messages, "messages"),
        ] {
            let value = serde_json::to_value(kind).expect("should serialize");
            expect_that!(&value, eq(&json!(expected)));
        }
    }
}

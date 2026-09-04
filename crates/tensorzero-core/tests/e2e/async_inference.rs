// Modified by Delta-AI under Apache 2.0
//! E2E tests for the async inference API:
//! `POST /v1/chat/completions/async` (and `/v1/responses/async`, `/v1/messages/async`),
//! `GET /v1/async_tasks/{task_id}`, and `GET /v1/async_tasks/{task_id}/stream`.
//!
//! These tests require a gateway started with `[gateway.async_inference] enabled = true`
//! plus `TENSORZERO_POSTGRES_URL` and `TENSORZERO_VALKEY_URL`, e.g.:
//!
//! ```bash
//! cargo run --bin gateway --features e2e_tests -- \
//!   --config-file "tensorzero-core/tests/e2e/config/{tensorzero,object-storage-disabled,async-inference}.*.toml"
//! ```
//!
//! Against a gateway without async inference enabled, every test here skips.

use std::time::{Duration, Instant};

use googletest::prelude::*;
use reqwest::{Client, StatusCode};
use reqwest_sse_stream::{Event, RequestBuilderExt};
use serde_json::{Value, json};
use tensorzero_core::endpoints::openai_compatible::async_inference_types::{
    AsyncInferenceLaunchResponse, AsyncTaskStatusResponse,
};
use tokio_stream::StreamExt;
use uuid::Uuid;

use crate::common::get_gateway_endpoint;
use tensorzero_core::db::clickhouse::test_helpers::{
    get_clickhouse, select_chat_inference_clickhouse,
};

/// Model referenced by the async inference tests; served by the dummy provider.
const TEST_MODEL: &str = "tensorzero::model_name::dummy::good";

const POLL_INTERVAL: Duration = Duration::from_millis(200);
const TERMINAL_TIMEOUT: Duration = Duration::from_secs(60);
const STREAM_TIMEOUT: Duration = Duration::from_secs(60);

/// Probes whether the running gateway has async inference enabled by fetching
/// the status of a random task ID: an enabled gateway answers 404 (task not
/// found), a disabled one answers 500 with a "not enabled" config error.
async fn async_inference_enabled(client: &Client) -> bool {
    let probe_path = format!("/v1/async_tasks/{}", Uuid::now_v7());
    let probe = client
        .get(get_gateway_endpoint(&probe_path))
        .send()
        .await
        .expect("failed to reach the gateway for the async inference probe");
    match probe.status() {
        StatusCode::NOT_FOUND => true,
        StatusCode::INTERNAL_SERVER_ERROR => {
            let body = probe.text().await.expect("probe body should read");
            assert!(
                body.contains("Async inference is not enabled"),
                "unexpected 500 from async task probe: {body}"
            );
            false
        }
        status => panic!("unexpected status from async task probe: {status}"),
    }
}

/// Skips the current test when the gateway under test does not have
/// `[gateway.async_inference] enabled = true` (e.g. the default `run-e2e`
/// gateway), the same way `skip_for_postgres!` skips unsupported backends.
macro_rules! skip_unless_async_inference {
    ($client:expr) => {
        if !async_inference_enabled(&$client).await {
            println!(
                "Skipping: the gateway was not started with `gateway.async_inference.enabled = true`"
            );
            return;
        }
    };
}

/// Submits an async inference request, asserting the `202 Accepted` launch response.
async fn submit_async(client: &Client, path: &str, body: &Value) -> AsyncInferenceLaunchResponse {
    let response = client
        .post(get_gateway_endpoint(path))
        .json(body)
        .send()
        .await
        .expect("async submit request should reach the gateway");
    assert_that!(
        response.status(),
        eq(StatusCode::ACCEPTED),
        "submit to {path} should be accepted"
    );
    response
        .json()
        .await
        .expect("async submit response should deserialize")
}

/// Fetches the current status of a task, asserting a `200 OK` typed response.
async fn get_task_status(client: &Client, task_id: Uuid) -> AsyncTaskStatusResponse {
    let path = format!("/v1/async_tasks/{task_id}");
    let response = client
        .get(get_gateway_endpoint(&path))
        .send()
        .await
        .expect("status request should reach the gateway");
    assert_that!(response.status(), eq(StatusCode::OK));
    response
        .json()
        .await
        .expect("task status response should deserialize")
}

/// Polls the status endpoint until the task reaches a terminal state
/// (completed / failed / cancelled), failing the test on timeout.
async fn wait_for_terminal_status(client: &Client, task_id: Uuid) -> AsyncTaskStatusResponse {
    let deadline = Instant::now() + TERMINAL_TIMEOUT;
    loop {
        let status = get_task_status(client, task_id).await;
        match status {
            AsyncTaskStatusResponse::Completed { .. }
            | AsyncTaskStatusResponse::Failed { .. }
            | AsyncTaskStatusResponse::Cancelled { .. } => return status,
            AsyncTaskStatusResponse::Queued { .. } | AsyncTaskStatusResponse::Running { .. } => {}
        }
        assert!(
            Instant::now() < deadline,
            "task {task_id} did not reach a terminal state within {TERMINAL_TIMEOUT:?}"
        );
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

fn chat_completions_body(model: &str) -> Value {
    json!({
        "model": model,
        "messages": [{"role": "user", "content": "Hello, world!"}],
    })
}

/// Asserts the ClickHouse row for `inference_id` carries the tags the async
/// worker stamps via `extra_internal_tags`: `tensorzero::async` and
/// `tensorzero::async_task_id`. Guards against regressions where worker tags
/// land in the client-visible map and `inference()` rejects the task.
async fn expect_async_tags_on_inference_row(inference_id: Uuid, task_id: Uuid) {
    let clickhouse = get_clickhouse().await;
    let row = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .expect("async-executed inference should be written to ClickHouse");
    let expected_task_id = task_id.to_string();
    expect_that!(
        row["tags"]["tensorzero::async"].as_str(),
        some(eq("true")),
        "inference row should be tagged `tensorzero::async`: {row}"
    );
    expect_that!(
        row["tags"]["tensorzero::async_task_id"].as_str(),
        some(eq(expected_task_id.as_str())),
        "inference row should be tagged with the durable task id: {row}"
    );
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_async_chat_completions_submit_and_complete() {
    let client = Client::new();
    skip_unless_async_inference!(client);

    let launch = submit_async(
        &client,
        "/v1/chat/completions/async",
        &chat_completions_body(TEST_MODEL),
    )
    .await;

    let final_status = wait_for_terminal_status(&client, launch.task_id).await;
    let AsyncTaskStatusResponse::Completed { task_id, response } = final_status else {
        panic!("expected the task to complete, got {final_status:?}");
    };
    assert_that!(task_id, eq(launch.task_id));
    expect_that!(
        response["object"].as_str(),
        some(eq("chat.completion")),
        "completed response should have the sync chat completions shape"
    );
    expect_that!(
        response["choices"][0]["message"]["role"].as_str(),
        some(eq("assistant"))
    );
    let content = response["choices"][0]["message"]["content"]
        .as_str()
        .expect("completed response should carry string content");
    expect_that!(content, not(eq("")));
    expect_that!(
        response["usage"].is_object(),
        eq(true),
        "completed response should carry usage"
    );

    let inference_id: Uuid = response["id"]
        .as_str()
        .expect("completed response should carry an inference id")
        .parse()
        .expect("inference id should be a UUID");
    expect_async_tags_on_inference_row(inference_id, task_id).await;
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_async_chat_completions_stream_matches_sync_shape() {
    let client = Client::new();
    skip_unless_async_inference!(client);

    // Submit via the `/openai/v1` alias to cover it too.
    let launch = submit_async(
        &client,
        "/openai/v1/chat/completions/async",
        &chat_completions_body(TEST_MODEL),
    )
    .await;

    let stream_path = format!("/v1/async_tasks/{}/stream", launch.task_id);
    let mut events = client
        .get(get_gateway_endpoint(&stream_path))
        .header("Accept", "text/event-stream")
        .eventsource()
        .await
        .expect("stream request should succeed");

    let mut chunks = vec![];
    let mut saw_done = false;
    let collect = async {
        while let Some(event) = events.next().await {
            let event = event.expect("SSE event should be valid");
            match event {
                Event::Open => continue,
                Event::Message(message) => {
                    if message.data == "[DONE]" {
                        saw_done = true;
                        break;
                    }
                    chunks.push(message.data);
                }
            }
        }
    };
    tokio::time::timeout(STREAM_TIMEOUT, collect)
        .await
        .expect("async task stream should terminate with `[DONE]`");

    assert_that!(
        saw_done,
        eq(true),
        "async task stream should end with `[DONE]` like the sync streaming API"
    );
    assert_that!(
        chunks.len(),
        ge(1),
        "async task stream should yield at least one chunk"
    );

    let mut streamed_content = String::new();
    for chunk in &chunks {
        let parsed: Value = serde_json::from_str(chunk).expect("chunk should be valid JSON");
        expect_that!(
            parsed["object"].as_str(),
            some(eq("chat.completion.chunk")),
            "chunk should have the sync streaming shape: {parsed}"
        );
        if let Some(content) = parsed["choices"][0]["delta"]["content"].as_str() {
            streamed_content.push_str(content);
        }
    }
    expect_that!(
        streamed_content.as_str(),
        not(eq("")),
        "streamed chunks should carry content"
    );

    // The status endpoint should agree that the task completed, and its final
    // response should carry the same content the stream produced.
    let final_status = wait_for_terminal_status(&client, launch.task_id).await;
    let AsyncTaskStatusResponse::Completed { response, .. } = final_status else {
        panic!("expected the streamed task to complete, got {final_status:?}");
    };
    let completed_content = response["choices"][0]["message"]["content"]
        .as_str()
        .expect("completed response should carry string content");
    expect_that!(
        completed_content,
        eq(streamed_content.as_str()),
        "streamed content should match the final response content"
    );
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_async_messages_anthropic_submit_and_complete() {
    let client = Client::new();
    skip_unless_async_inference!(client);

    // The Anthropic messages API takes plain model names, like the sync endpoint.
    let body = json!({
        "model": "dummy::good",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello, world!"}],
    });
    let launch = submit_async(&client, "/v1/messages/async", &body).await;

    let final_status = wait_for_terminal_status(&client, launch.task_id).await;
    let AsyncTaskStatusResponse::Completed { task_id, response } = final_status else {
        panic!("expected the task to complete, got {final_status:?}");
    };
    assert_that!(task_id, eq(launch.task_id));
    expect_that!(
        response["type"].as_str(),
        some(eq("message")),
        "completed response should have the Anthropic messages shape"
    );
    expect_that!(response["role"].as_str(), some(eq("assistant")));
    expect_that!(response["content"][0]["type"].as_str(), some(eq("text")));
    let text = response["content"][0]["text"]
        .as_str()
        .expect("completed response should carry a text block");
    expect_that!(text, not(eq("")));

    let inference_id: Uuid = response["id"]
        .as_str()
        .and_then(|id| id.strip_prefix("msg_"))
        .expect("completed response should carry a `msg_<uuid>` id")
        .parse()
        .expect("inference id should be a UUID");
    expect_async_tags_on_inference_row(inference_id, task_id).await;
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_async_chat_completions_unknown_model_fails() {
    let client = Client::new();
    skip_unless_async_inference!(client);

    // Submit-time validation does not resolve model names, so an unknown model
    // is accepted and then fails during execution, surfacing as a failed task.
    let launch = submit_async(
        &client,
        "/v1/chat/completions/async",
        &chat_completions_body("tensorzero::model_name::definitely_missing_model"),
    )
    .await;

    let final_status = wait_for_terminal_status(&client, launch.task_id).await;
    let AsyncTaskStatusResponse::Failed { task_id, error } = final_status else {
        panic!("expected the task to fail, got {final_status:?}");
    };
    assert_that!(task_id, eq(launch.task_id));
    let error = error.expect("a failed task should carry an error payload");
    let message = error
        .pointer("/error/message")
        .or_else(|| error.get("message"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    expect_that!(
        message,
        not(eq("")),
        "the task error should carry a message: {error}"
    );
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_async_task_unknown_id_returns_404() {
    let client = Client::new();
    skip_unless_async_inference!(client);

    let task_id = Uuid::now_v7();
    let status_response = client
        .get(get_gateway_endpoint(&format!("/v1/async_tasks/{task_id}")))
        .send()
        .await
        .expect("status request should reach the gateway");
    assert_that!(status_response.status(), eq(StatusCode::NOT_FOUND));

    let stream_response = client
        .get(get_gateway_endpoint(&format!(
            "/v1/async_tasks/{task_id}/stream"
        )))
        .send()
        .await
        .expect("stream request should reach the gateway");
    assert_that!(stream_response.status(), eq(StatusCode::NOT_FOUND));
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_async_chat_completions_invalid_body_returns_400() {
    let client = Client::new();
    skip_unless_async_inference!(client);

    // A body that fails synchronous request validation (no `model`) is
    // rejected at submit time rather than enqueued.
    let response = client
        .post(get_gateway_endpoint("/v1/chat/completions/async"))
        .json(&json!({"messages": [{"role": "user", "content": "Hello"}]}))
        .send()
        .await
        .expect("submit request should reach the gateway");
    assert_that!(response.status(), eq(StatusCode::BAD_REQUEST));
}

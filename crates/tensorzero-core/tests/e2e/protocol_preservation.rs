// Modified by Delta-AI under Apache 2.0
//! End-to-end coverage for inbound protocol preservation (Delta-AI fork).
//!
//! A request that arrives at `/openai/v1/responses` must go out to the
//! provider over the Responses API, and a request that arrives at
//! `/openai/v1/chat/completions` must go out over chat completions, even when
//! the provider's configured `api_type` differs. Only single-protocol
//! providers fall back to conversion.
//!
//! Upstream is a tiny mock OpenAI-compatible server that records the request
//! bodies it receives, so these tests travel with the repo and do not need
//! live vendor keys.

use std::collections::HashMap;
use std::future::IntoFuture;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::Response;
use axum::routing::post;
use axum::{Json, Router};
use googletest::prelude::*;
use http_body_util::BodyExt;
use reqwest::StatusCode;
use serde_json::{Value, json};
use tensorzero::ClientExt;
use tensorzero_core::endpoints::openai_compatible::OpenAIStructuredJson;
use tensorzero_core::endpoints::openai_compatible::chat_completions::chat_completions_handler;
use tensorzero_core::endpoints::openai_compatible::responses::responses_handler;

type RecordedRequests = Arc<Mutex<HashMap<String, Vec<Value>>>>;

/// Spawn a mock OpenAI-compatible server that records every request body per
/// path and replies with canned chat-completions / responses payloads.
async fn make_recording_openai_server() -> (
    SocketAddr,
    RecordedRequests,
    tokio::sync::oneshot::Sender<()>,
) {
    let addr = SocketAddr::from(([127, 0, 0, 1], 0));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .unwrap_or_else(|e| panic!("Failed to bind to {addr}: {e}"));
    let real_addr = listener.local_addr().unwrap();

    let recorded: RecordedRequests = Arc::new(Mutex::new(HashMap::new()));

    fn record(recorded: &RecordedRequests, path: &str, body: Value) {
        recorded
            .lock()
            .expect("recorded requests lock")
            .entry(path.to_string())
            .or_default()
            .push(body);
    }

    let chat_recorded = recorded.clone();
    let responses_recorded = recorded.clone();
    let app = Router::new()
        .route(
            "/v1/chat/completions",
            post(move |Json(body): Json<Value>| {
                let recorded = chat_recorded.clone();
                async move {
                    record(&recorded, "/v1/chat/completions", body);
                    Json(json!({
                        "id": "chatcmpl-mock",
                        "object": "chat.completion",
                        "created": 1,
                        "model": "gpt-4.1-mini",
                        "choices": [{
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "chat OK"}
                        }],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
                    }))
                }
            }),
        )
        .route(
            "/v1/responses",
            post(move |Json(body): Json<Value>| {
                let recorded = responses_recorded.clone();
                async move {
                    record(&recorded, "/v1/responses", body);
                    Json(json!({
                        "id": "resp_mock",
                        "object": "response",
                        "created_at": 1,
                        "status": "completed",
                        "model": "gpt-4.1-mini",
                        "output": [{
                            "id": "msg_mock",
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "responses OK", "annotations": []}]
                        }],
                        "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
                    }))
                }
            }),
        );

    let (send, recv) = tokio::sync::oneshot::channel::<()>();
    #[expect(clippy::disallowed_methods, reason = "test code")]
    tokio::spawn(
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = recv.await;
            })
            .into_future(),
    );

    (real_addr, recorded, send)
}

fn gateway_config(addr: &SocketAddr) -> String {
    format!(
        r#"
[models.proto_mock]
routing = ["mock-openai"]

[models.proto_mock.providers.mock-openai]
type = "openai"
api_base = "http://{addr}/v1/"
api_key_location = "none"
model_name = "gpt-4.1-mini"

[models.proto_mock_responses]
routing = ["mock-openai-responses"]

[models.proto_mock_responses.providers.mock-openai-responses]
type = "openai"
api_base = "http://{addr}/v1/"
api_key_location = "none"
model_name = "gpt-4.1-mini"
api_type = "responses"

[models.proto_mock_flagged]
routing = ["mock-openai-flagged"]

[models.proto_mock_flagged.providers.mock-openai-flagged]
type = "openai"
api_base = "http://{addr}/v1/"
api_key_location = "none"
model_name = "gpt-4.1-mini"
responses_structured_output_fallback_to_chat = true
"#
    )
}

async fn json_of(response: Response) -> (StatusCode, Value) {
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: Value =
        serde_json::from_slice(&bytes).unwrap_or_else(|_| json!({"raw": bytes.len()}));
    (status, body)
}

fn recorded_bodies(recorded: &RecordedRequests, path: &str) -> Vec<Value> {
    recorded
        .lock()
        .expect("recorded requests lock")
        .get(path)
        .cloned()
        .unwrap_or_default()
}

/// The gateway's OpenAI-compatible endpoints default the response cache to
/// ON, and the e2e ClickHouse database persists across local runs. A unique
/// input per run keeps these tests hermetic: they always exercise the
/// provider path (whose outbound shape they assert), never a cache hit.
fn unique_input(prefix: &str) -> String {
    format!("{prefix}-{}", uuid::Uuid::now_v7())
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn responses_inbound_stays_responses_outbound() {
    let (addr, recorded, _shutdown) = make_recording_openai_server().await;
    let client =
        tensorzero::test_helpers::make_embedded_gateway_with_config(&gateway_config(&addr)).await;
    let state = client.get_app_state_data().unwrap().load_latest();

    // The provider is chat-configured; an inbound Responses request must
    // still go out over the Responses API, with `text`/`reasoning` preserved.
    let input = unique_input("Hello");
    let (status, body) = json_of(
        responses_handler(
            State(state.clone()),
            None,
            HeaderMap::new(),
            OpenAIStructuredJson(
                serde_json::from_value(json!({
                    "model": "proto_mock",
                    "input": input,
                    "text": {"format": {"type": "text"}, "verbosity": "low"},
                    "reasoning": {"effort": "low"}
                }))
                .unwrap(),
            ),
        )
        .await
        .unwrap(),
    )
    .await;
    assert_that!(status, eq(StatusCode::OK));
    expect_that!(body["object"].as_str(), some(eq("response")));

    let responses_bodies = recorded_bodies(&recorded, "/v1/responses");
    assert_that!(responses_bodies.len(), eq(1));
    let upstream = &responses_bodies[0];
    // Responses wire shape: `input`, no `messages`.
    expect_that!(upstream.get("input").is_some(), eq(true));
    expect_that!(upstream.get("messages").is_none(), eq(true));
    // Inbound `text`/`reasoning` fields survive to the outbound request.
    expect_that!(upstream["reasoning"]["effort"].as_str(), some(eq("low")));
    expect_that!(upstream["text"]["verbosity"].as_str(), some(eq("low")));
    expect_that!(
        recorded_bodies(&recorded, "/v1/chat/completions").len(),
        eq(0)
    );
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn chat_inbound_stays_chat_outbound() {
    let (addr, recorded, _shutdown) = make_recording_openai_server().await;
    let client =
        tensorzero::test_helpers::make_embedded_gateway_with_config(&gateway_config(&addr)).await;
    let state = client.get_app_state_data().unwrap().load_latest();

    // The provider is responses-configured; an inbound chat request must be
    // downgraded to chat completions outbound.
    let input = unique_input("Hello");
    let (status, body) = json_of(
        chat_completions_handler(
            State(state.clone()),
            None,
            HeaderMap::new(),
            OpenAIStructuredJson(
                serde_json::from_value(json!({
                    "model": "proto_mock_responses",
                    "messages": [{"role": "user", "content": input}]
                }))
                .unwrap(),
            ),
        )
        .await
        .unwrap(),
    )
    .await;
    assert_that!(status, eq(StatusCode::OK));
    expect_that!(body["object"].as_str(), some(eq("chat.completion")));

    let chat_bodies = recorded_bodies(&recorded, "/v1/chat/completions");
    assert_that!(chat_bodies.len(), eq(1));
    let upstream = &chat_bodies[0];
    // Chat wire shape: `messages`, no `input`.
    expect_that!(upstream.get("messages").is_some(), eq(true));
    expect_that!(upstream.get("input").is_none(), eq(true));
    expect_that!(recorded_bodies(&recorded, "/v1/responses").len(), eq(0));
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn matching_inbound_and_configured_protocols_still_work() {
    let (addr, recorded, _shutdown) = make_recording_openai_server().await;
    let client =
        tensorzero::test_helpers::make_embedded_gateway_with_config(&gateway_config(&addr)).await;
    let state = client.get_app_state_data().unwrap().load_latest();

    // Chat inbound against a chat-configured provider (regression: unchanged).
    let input = unique_input("Hello");
    let (status, _) = json_of(
        chat_completions_handler(
            State(state.clone()),
            None,
            HeaderMap::new(),
            OpenAIStructuredJson(
                serde_json::from_value(json!({
                    "model": "proto_mock",
                    "messages": [{"role": "user", "content": input}]
                }))
                .unwrap(),
            ),
        )
        .await
        .unwrap(),
    )
    .await;
    assert_that!(status, eq(StatusCode::OK));

    // Responses inbound against a responses-configured provider.
    let (status, _) = json_of(
        responses_handler(
            State(state.clone()),
            None,
            HeaderMap::new(),
            OpenAIStructuredJson(
                serde_json::from_value(json!({
                    "model": "proto_mock_responses",
                    "input": unique_input("Hello")
                }))
                .unwrap(),
            ),
        )
        .await
        .unwrap(),
    )
    .await;
    assert_that!(status, eq(StatusCode::OK));

    expect_that!(
        recorded_bodies(&recorded, "/v1/chat/completions").len(),
        eq(1)
    );
    expect_that!(recorded_bodies(&recorded, "/v1/responses").len(), eq(1));
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn flagged_provider_downgrades_structured_responses_to_chat() {
    let (addr, recorded, _shutdown) = make_recording_openai_server().await;
    let client =
        tensorzero::test_helpers::make_embedded_gateway_with_config(&gateway_config(&addr)).await;
    let state = client.get_app_state_data().unwrap().load_latest();

    // A provider flagged `responses_structured_output_fallback_to_chat`
    // (e.g. Alibaba Bailian, whose /responses ignores text.format): an
    // inbound Responses request with a structured-output format must go out
    // over chat completions with `response_format` intact.
    let input = unique_input("Hello");
    let (status, body) = json_of(
        responses_handler(
            State(state.clone()),
            None,
            HeaderMap::new(),
            OpenAIStructuredJson(
                serde_json::from_value(json!({
                    "model": "proto_mock_flagged",
                    "input": input,
                    "text": {"format": {"type": "json_schema", "name": "person", "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                        "additionalProperties": false
                    }}}
                }))
                .unwrap(),
            ),
        )
        .await
        .unwrap(),
    )
    .await;
    assert_that!(status, eq(StatusCode::OK));
    // The gateway still answers in the inbound (Responses) wire shape.
    expect_that!(body["object"].as_str(), some(eq("response")));

    let chat_bodies = recorded_bodies(&recorded, "/v1/chat/completions");
    assert_that!(chat_bodies.len(), eq(1));
    let upstream = &chat_bodies[0];
    // Chat wire shape with the schema preserved as `response_format`.
    expect_that!(upstream.get("messages").is_some(), eq(true));
    expect_that!(
        upstream["response_format"]["type"].as_str(),
        some(eq("json_schema"))
    );
    expect_that!(recorded_bodies(&recorded, "/v1/responses").len(), eq(0));

    // Without a structured format, the same flagged provider stays on
    // Responses (streaming/reasoning keep working there).
    let (status, _) = json_of(
        responses_handler(
            State(state.clone()),
            None,
            HeaderMap::new(),
            OpenAIStructuredJson(
                serde_json::from_value(json!({
                    "model": "proto_mock_flagged",
                    "input": unique_input("Hello")
                }))
                .unwrap(),
            ),
        )
        .await
        .unwrap(),
    )
    .await;
    assert_that!(status, eq(StatusCode::OK));
    expect_that!(recorded_bodies(&recorded, "/v1/responses").len(), eq(1));
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn responses_inbound_downgrades_for_single_protocol_provider() {
    // The per-request protocol preference is a hint that dual-protocol
    // providers honor: single-protocol providers (here: the dummy provider,
    // which speaks neither OpenAI protocol) must keep working when the
    // inbound request arrived on `/openai/v1/responses` — they ignore the
    // hint and speak their own protocol (downgrade conversion).
    let config = r#"
[models.proto_dummy]
routing = ["dummy-provider"]

[models.proto_dummy.providers.dummy-provider]
type = "dummy"
model_name = "good"
"#;
    let client = tensorzero::test_helpers::make_embedded_gateway_with_config(config).await;
    let state = client.get_app_state_data().unwrap().load_latest();

    let (status, body) = json_of(
        responses_handler(
            State(state.clone()),
            None,
            HeaderMap::new(),
            OpenAIStructuredJson(
                serde_json::from_value(json!({
                    "model": "proto_dummy",
                    "input": unique_input("Hello"),
                }))
                .unwrap(),
            ),
        )
        .await
        .unwrap(),
    )
    .await;
    assert_that!(status, eq(StatusCode::OK));
    // The gateway still answers in the inbound (Responses) wire shape.
    expect_that!(body["object"].as_str(), some(eq("response")));
    expect_that!(body["status"].as_str(), some(eq("completed")));
    let output_text = body["output"][0]["content"][0]["text"].as_str();
    expect_that!(output_text.is_some(), eq(true));
}

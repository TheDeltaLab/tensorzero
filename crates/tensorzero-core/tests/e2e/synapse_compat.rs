// Modified by Delta-AI under Apache 2.0
//! End-to-end coverage for the Synapse compatibility layer.
//!
//! Upstream LLMs are the Dummy provider (`dummy::*`). These tests call the
//! OpenAI-compatible / Anthropic / internal handlers through an embedded
//! gateway so they travel with the repo and do not need live vendor keys.
#![expect(clippy::print_stdout)]

use std::future::IntoFuture;
use std::net::SocketAddr;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::extract::{Query, State};
use axum::http::HeaderMap;
use axum::response::Response;
use axum::routing::get;
use chrono::{TimeZone, Utc};
use googletest::prelude::*;
use http_body_util::BodyExt;
use reqwest::StatusCode;
use serde_json::{Value, json};
use tensorzero::ClientExt;
use tensorzero_core::cache::CacheEnabledMode;
use tensorzero_core::db::clickhouse::query_builder::{
    InferenceFilter, TagComparisonOperator, TagFilter,
};
use tensorzero_core::db::delegating_connection::DelegatingDatabaseConnection;
use tensorzero_core::db::inferences::{InferenceQueries, ListInferencesParams};
use tensorzero_core::db::model_inferences::ModelInferenceQueries;
use tensorzero_core::db::test_helpers::TestDatabaseHelpers;
use tensorzero_core::endpoints::internal::synapse::{
    SynapseTimeRangeQuery, analytics_handler, balances_handler, usage_export_handler,
};
use tensorzero_core::endpoints::openai_compatible::OpenAIStructuredJson;
use tensorzero_core::endpoints::openai_compatible::anthropic_messages::messages_handler;
use tensorzero_core::endpoints::openai_compatible::build_openai_compatible_routes;
use tensorzero_core::endpoints::openai_compatible::chat_completions::chat_completions_handler;
use tensorzero_core::endpoints::openai_compatible::completions::completions_handler;
use tensorzero_core::endpoints::openai_compatible::embeddings::embeddings_handler;
use tensorzero_core::endpoints::openai_compatible::rerank::rerank_handler;
use tensorzero_core::endpoints::openai_compatible::responses::responses_handler;
use tensorzero_core::endpoints::openai_compatible::synapse::{
    LONG_AUDIO_EVAL_TIMEOUT, resolve_cache_options, resolve_openai_compatible_model,
};
use tensorzero_core::endpoints::openai_compatible::types::chat_completions::OpenAICompatibleParams;
use tensorzero_core::model::SHORTHAND_MODEL_PREFIXES;
use tensorzero_core::stored_inference::StoredInferenceDatabase;
use tensorzero_core::test_helpers::get_e2e_config;
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

const MEGUMIN: &str = "Megumin gleefully chanted her spell";
const WALLY: &str = "Wally, the golden retriever";

fn chat_body(model: &str) -> OpenAIStructuredJson<OpenAICompatibleParams> {
    OpenAIStructuredJson(
        serde_json::from_value(json!({
            "model": model,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": false,
        }))
        .unwrap(),
    )
}

async fn json_of(response: Response) -> (StatusCode, HeaderMap, Value) {
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: Value =
        serde_json::from_slice(&bytes).unwrap_or_else(|_| json!({"raw": bytes.len()}));
    (status, headers, body)
}

async fn text_of(response: Response) -> (StatusCode, HeaderMap, String) {
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (
        status,
        headers,
        String::from_utf8(bytes.to_vec()).unwrap_or_default(),
    )
}

fn served_by(headers: &HeaderMap) -> &str {
    headers
        .get("x-synapse-served-by")
        .unwrap()
        .to_str()
        .unwrap()
}

fn fallback_count(headers: &HeaderMap) -> &str {
    headers
        .get("x-synapse-fallback-count")
        .unwrap()
        .to_str()
        .unwrap()
}

async fn live_gateway() -> Option<reqwest::Client> {
    let client = reqwest::Client::new();
    let ok = match tokio::time::timeout(
        Duration::from_secs(2),
        client.get(get_gateway_endpoint("/status")).send(),
    )
    .await
    {
        Ok(Ok(response)) => response.status().is_success(),
        _ => false,
    };
    ok.then_some(client)
}

/// Tiny PCM WAV so Dummy can fetch `input_audio` HTTP URLs without hitting the public internet.
fn tiny_wav() -> Vec<u8> {
    let mut wav = Vec::new();
    wav.extend(b"RIFF");
    wav.extend(&(38u32).to_le_bytes());
    wav.extend(b"WAVEfmt ");
    wav.extend(&16u32.to_le_bytes());
    wav.extend(&1u16.to_le_bytes());
    wav.extend(&1u16.to_le_bytes());
    wav.extend(&8000u32.to_le_bytes());
    wav.extend(&16000u32.to_le_bytes());
    wav.extend(&2u16.to_le_bytes());
    wav.extend(&16u16.to_le_bytes());
    wav.extend(b"data");
    wav.extend(&2u32.to_le_bytes());
    wav.extend(&0u16.to_le_bytes());
    wav
}

async fn serve_tiny_wav() -> (String, tokio::sync::oneshot::Sender<()>) {
    let listener = tokio::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
        .await
        .unwrap();
    let addr = listener.local_addr().unwrap();
    let wav = tiny_wav();
    let app = Router::new().route(
        "/eval.wav",
        get(move || {
            let wav = wav.clone();
            async move {
                Response::builder()
                    .header(axum::http::header::CONTENT_TYPE, "audio/wav")
                    .body(Body::from(wav))
                    .unwrap()
            }
        }),
    );
    let (send, recv) = tokio::sync::oneshot::channel::<()>();
    #[expect(clippy::disallowed_methods)]
    tokio::spawn(
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = recv.await;
            })
            .into_future(),
    );
    (format!("http://{addr}/eval.wav"), send)
}

fn sse_data_values(body: &str) -> Vec<Value> {
    body.split("\n\n")
        .filter_map(|block| {
            block.lines().find_map(|line| {
                let data = line
                    .strip_prefix("data: ")
                    .or_else(|| line.strip_prefix("data:"))?;
                if data == "[DONE]" {
                    None
                } else {
                    serde_json::from_str(data).ok()
                }
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// A — contract
// ---------------------------------------------------------------------------

#[gtest]
fn synapse_routes_register_v1_and_anthropic_aliases() {
    let paths: Vec<_> = build_openai_compatible_routes()
        .routes
        .iter()
        .map(|(path, _)| *path)
        .collect();
    for path in [
        "/v1/chat/completions",
        "/openai/v1/chat/completions",
        "/v1/embeddings",
        "/v1/completions",
        "/v1/responses",
        "/v1/rerank",
        "/v1/reranks",
        "/v1/messages",
        "/openai/v1/messages",
        "/anthropic/v1/messages",
    ] {
        expect_that!(paths.contains(&path), eq(true));
    }
}

#[gtest]
fn synapse_domestic_shorthands_are_registered() {
    for prefix in ["alibaba::", "siliconflow::", "volcengine::"] {
        expect_that!(SHORTHAND_MODEL_PREFIXES.contains(&prefix), eq(true));
    }
    expect_that!(
        resolve_openai_compatible_model("qwen-plus", Some("alibaba")).unwrap(),
        eq("alibaba::qwen-plus")
    );
    expect_that!(
        resolve_openai_compatible_model("Qwen/Qwen3-Embedding-4B", Some("siliconflow")).unwrap(),
        eq("siliconflow::Qwen/Qwen3-Embedding-4B")
    );
    expect_that!(
        resolve_openai_compatible_model("ep-xxx", Some("volcengine")).unwrap(),
        eq("volcengine::ep-xxx")
    );
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn synapse_bare_model_and_provider_header() {
    let client = tensorzero::test_helpers::make_embedded_gateway_no_config().await;
    let state = client.get_app_state_data().unwrap().load_latest();
    let mut headers = HeaderMap::new();
    headers.insert("x-synapse-provider", "dummy".parse().unwrap());
    headers.insert("x-synapse-request-id", "e2e-trace-a".parse().unwrap());
    let (status, headers, body) = json_of(
        chat_completions_handler(State(state), None, headers, chat_body("good"))
            .await
            .unwrap(),
    )
    .await;
    assert_that!(status, eq(StatusCode::OK));
    expect_that!(served_by(&headers), eq("dummy/good"));
    expect_that!(
        headers
            .get("x-synapse-request-id")
            .unwrap()
            .to_str()
            .unwrap(),
        eq("e2e-trace-a")
    );
    expect_that!(fallback_count(&headers), eq("0"));
    expect_that!(
        body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .contains(MEGUMIN),
        eq(true)
    );
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn tensorzero_headers_activate_synapse_compat_and_echo_both_prefixes() {
    let client = tensorzero::test_helpers::make_embedded_gateway_no_config().await;
    let state = client.get_app_state_data().unwrap().load_latest();
    let mut headers = HeaderMap::new();
    headers.insert("x-tensorzero-provider", "dummy".parse().unwrap());
    headers.insert("x-tensorzero-request-id", "tz-trace-a".parse().unwrap());
    let (status, headers, _) = json_of(
        chat_completions_handler(State(state), None, headers, chat_body("good"))
            .await
            .unwrap(),
    )
    .await;
    assert_that!(status, eq(StatusCode::OK));
    expect_that!(served_by(&headers), eq("dummy/good"));
    expect_that!(
        headers
            .get("x-tensorzero-served-by")
            .unwrap()
            .to_str()
            .unwrap(),
        eq("dummy/good")
    );
    expect_that!(
        headers
            .get("x-tensorzero-request-id")
            .unwrap()
            .to_str()
            .unwrap(),
        eq("tz-trace-a")
    );
    expect_that!(
        headers
            .get("x-synapse-request-id")
            .unwrap()
            .to_str()
            .unwrap(),
        eq("tz-trace-a")
    );
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn tensorzero_episode_and_tags_headers_persist() {
    let client = tensorzero::test_helpers::make_embedded_gateway_no_config().await;
    let state = client.get_app_state_data().unwrap().load_latest();
    let episode_id = Uuid::now_v7();
    let mut headers = HeaderMap::new();
    headers.insert("x-tensorzero-provider", "dummy".parse().unwrap());
    headers.insert(
        "x-tensorzero-episodes-id",
        episode_id.to_string().parse().unwrap(),
    );
    headers.insert(
        "x-tensorzero-tags",
        "env=prod,team=ml,canary".parse().unwrap(),
    );
    let (status, _, body) = json_of(
        chat_completions_handler(State(state), None, headers, chat_body("good"))
            .await
            .unwrap(),
    )
    .await;
    assert_that!(status, eq(StatusCode::OK));
    expect_that!(
        body["episode_id"].as_str().unwrap(),
        eq(episode_id.to_string().as_str())
    );

    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let config = get_e2e_config().await;
    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                episode_id: Some(&episode_id),
                limit: 10,
                ..Default::default()
            },
        )
        .await
        .expect("list inferences by episode header");
    let chat = inferences.iter().find_map(|inference| match inference {
        StoredInferenceDatabase::Chat(chat) => Some(chat),
        StoredInferenceDatabase::Json(_) => None,
    });
    let chat = chat.expect("expected chat inference for episode header");
    expect_that!(chat.episode_id, eq(episode_id));
    expect_that!(chat.tags.get("env").map(String::as_str), eq(Some("prod")));
    expect_that!(chat.tags.get("team").map(String::as_str), eq(Some("ml")));
    expect_that!(
        chat.tags.get("canary").map(String::as_str),
        eq(Some("true"))
    );
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn synapse_chat_persists_request_headers_and_filters_by_request_id() {
    let client = tensorzero::test_helpers::make_embedded_gateway_no_config().await;
    let state = client.get_app_state_data().unwrap().load_latest();
    let request_id = format!("e2e-obs-{}", Uuid::now_v7());
    let mut headers = HeaderMap::new();
    headers.insert("x-synapse-provider", "dummy".parse().unwrap());
    headers.insert("x-synapse-request-id", request_id.parse().unwrap());
    let (status, _, _) = json_of(
        chat_completions_handler(State(state), None, headers, chat_body("good"))
            .await
            .unwrap(),
    )
    .await;
    assert_that!(status, eq(StatusCode::OK));

    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let config = get_e2e_config().await;
    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                filters: Some(&InferenceFilter::Or {
                    children: vec![
                        InferenceFilter::Tag(TagFilter {
                            key: "tensorzero::synapse_request_id".to_string(),
                            value: request_id.clone(),
                            comparison_operator: TagComparisonOperator::Equal,
                        }),
                        InferenceFilter::Tag(TagFilter {
                            key: "tensorzero::provider_request_id".to_string(),
                            value: request_id.clone(),
                            comparison_operator: TagComparisonOperator::Equal,
                        }),
                    ],
                }),
                limit: 10,
                ..Default::default()
            },
        )
        .await
        .expect("list inferences by request id");
    let chat = inferences.iter().find_map(|inference| match inference {
        StoredInferenceDatabase::Chat(chat) => Some(chat),
        StoredInferenceDatabase::Json(_) => None,
    });
    let chat = chat.unwrap_or_else(|| panic!("expected chat inference for {request_id}"));
    expect_that!(
        chat.tags
            .get("tensorzero::synapse_request_id")
            .map(String::as_str),
        eq(Some(request_id.as_str()))
    );
    expect_that!(
        chat.tags.get("tensorzero::provider").map(String::as_str),
        eq(Some("dummy"))
    );
    expect_that!(
        chat.tags
            .get("tensorzero::header::x-tensorzero-request-id")
            .map(String::as_str),
        eq(Some(request_id.as_str()))
    );
    expect_that!(
        chat.tags
            .get("tensorzero::request_headers")
            .map(String::as_str)
            .unwrap_or("")
            .contains(&request_id),
        eq(true)
    );
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn synapse_completions_and_responses() {
    let client = tensorzero::test_helpers::make_embedded_gateway_no_config().await;
    let state = client.get_app_state_data().unwrap().load_latest();
    let (status, headers, body) = json_of(
        completions_handler(
            State(state.clone()),
            None,
            HeaderMap::new(),
            OpenAIStructuredJson(
                serde_json::from_value(json!({
                    "model": "dummy::good",
                    "prompt": "Hello",
                    "max_tokens": 16,
                }))
                .unwrap(),
            ),
        )
        .await
        .unwrap(),
    )
    .await;
    assert_that!(status, eq(StatusCode::OK));
    expect_that!(served_by(&headers), eq("dummy/good"));
    expect_that!(body["object"].as_str().unwrap(), eq("text_completion"));
    expect_that!(
        body["choices"][0]["text"]
            .as_str()
            .unwrap()
            .contains(MEGUMIN),
        eq(true)
    );

    let (status, _, body) = json_of(
        responses_handler(
            State(state),
            None,
            HeaderMap::new(),
            OpenAIStructuredJson(
                serde_json::from_value(json!({
                    "model": "dummy::good",
                    "input": "Hello",
                }))
                .unwrap(),
            ),
        )
        .await
        .unwrap(),
    )
    .await;
    assert_that!(status, eq(StatusCode::OK));
    expect_that!(body["object"].as_str().unwrap(), eq("response"));
    expect_that!(
        body["output"][0]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains(MEGUMIN),
        eq(true)
    );
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn synapse_live_v1_path_aliases() {
    let Some(http) = live_gateway().await else {
        println!("skip synapse_live_v1_path_aliases: gateway not reachable");
        return;
    };
    let payload = json!({
        "model": "dummy::good",
        "messages": [{"role": "user", "content": "Hi"}],
        "tensorzero::dryrun": true,
    });
    for path in ["/v1/chat/completions", "/openai/v1/chat/completions"] {
        let response = http
            .post(get_gateway_endpoint(path))
            .json(&payload)
            .send()
            .await
            .unwrap();
        assert_that!(response.status(), eq(StatusCode::OK));
    }
}

// ---------------------------------------------------------------------------
// B — routing
// ---------------------------------------------------------------------------

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn synapse_alias_failover_and_fallback_false() {
    let config = r#"
[model_aliases.alias_failover]
task = "chat"
targets = [
  { provider = "dummy", model = "error" },
  { provider = "dummy", model = "good" },
]
"#;
    let client = tensorzero::test_helpers::make_embedded_gateway_with_config(config).await;
    let (status, headers, _) = json_of(
        chat_completions_handler(
            State(client.get_app_state_data().unwrap().load_latest()),
            None,
            HeaderMap::new(),
            chat_body("alias_failover"),
        )
        .await
        .unwrap(),
    )
    .await;
    assert_that!(status, eq(StatusCode::OK));
    expect_that!(served_by(&headers), eq("dummy/good"));
    expect_that!(fallback_count(&headers), eq("1"));

    let mut headers = HeaderMap::new();
    headers.insert("x-synapse-fallback", "false".parse().unwrap());
    let (status, _, _) = json_of(
        chat_completions_handler(
            State(client.get_app_state_data().unwrap().load_latest()),
            None,
            headers,
            chat_body("alias_failover"),
        )
        .await
        .unwrap(),
    )
    .await;
    assert_that!(status, eq(StatusCode::BAD_GATEWAY));
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn synapse_min_tokens_per_sec_alias_still_serves() {
    let config = r#"
[model_aliases.gated]
task = "chat"
min_tokens_per_sec = 10.0
targets = [{ provider = "dummy", model = "good" }]
"#;
    let client = tensorzero::test_helpers::make_embedded_gateway_with_config(config).await;
    let (status, headers, _) = json_of(
        chat_completions_handler(
            State(client.get_app_state_data().unwrap().load_latest()),
            None,
            HeaderMap::new(),
            chat_body("gated"),
        )
        .await
        .unwrap(),
    )
    .await;
    assert_that!(status, eq(StatusCode::OK));
    expect_that!(served_by(&headers), eq("dummy/good"));
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn synapse_rerank_dummy() {
    let client = tensorzero::test_helpers::make_embedded_gateway_no_config().await;
    let marker = format!("obs-rerank-{}", Uuid::now_v7());
    let mut headers = HeaderMap::new();
    headers.insert("x-synapse-provider", "dummy".parse().unwrap());
    let (status, headers, body) = json_of(
        rerank_handler(
            State(client.get_app_state_data().unwrap().load_latest()),
            headers,
            OpenAIStructuredJson(
                serde_json::from_value(json!({
                    "model": "qwen3-rerank",
                    "query": marker,
                    "documents": [
                        "Paris is the capital of France.",
                        "London is the capital of England."
                    ],
                    "top_n": 1
                }))
                .unwrap(),
            ),
        )
        .await
        .unwrap(),
    )
    .await;
    assert_that!(status, eq(StatusCode::OK));
    expect_that!(served_by(&headers), eq("dummy/qwen3-rerank"));
    expect_that!(body["results"][0]["index"].as_u64().unwrap(), eq(0));
    expect_that!(body["results"].as_array().unwrap().len(), eq(1));
    assert_standalone_inference_recorded(&marker, "dummy::qwen3-rerank", "rerank", "Reranked")
        .await;
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn synapse_volcengine_input_audio_http_url() {
    let (audio_url, _shutdown) = serve_tiny_wav().await;
    let config = r#"
[object_storage]
type = "disabled"
"#;
    let client = tensorzero::test_helpers::make_embedded_gateway_with_config(config).await;
    let (status, _, body) = json_of(
        chat_completions_handler(
            State(client.get_app_state_data().unwrap().load_latest()),
            None,
            HeaderMap::new(),
            OpenAIStructuredJson(
                serde_json::from_value(json!({
                    "model": "dummy::good",
                    "messages": [{
                        "role": "user",
                        "content": [{
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_url,
                                "format": "wav"
                            }
                        }]
                    }],
                    "stream": false,
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
        body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .contains(MEGUMIN),
        eq(true)
    );
}

// ---------------------------------------------------------------------------
// C — protocol
// ---------------------------------------------------------------------------

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn synapse_anthropic_messages_non_streaming() {
    let client = tensorzero::test_helpers::make_embedded_gateway_no_config().await;
    let (status, headers, body) = json_of(
        messages_handler(
            State(client.get_app_state_data().unwrap().load_latest()),
            None,
            HeaderMap::new(),
            OpenAIStructuredJson(
                serde_json::from_value(json!({
                    "model": "dummy::good",
                    "max_tokens": 32,
                    "messages": [{"role": "user", "content": "Hello"}],
                }))
                .unwrap(),
            ),
        )
        .await
        .unwrap(),
    )
    .await;
    assert_that!(status, eq(StatusCode::OK));
    expect_that!(served_by(&headers), eq("dummy/good"));
    expect_that!(body["type"].as_str().unwrap(), eq("message"));
    expect_that!(body["role"].as_str().unwrap(), eq("assistant"));
    expect_that!(
        body["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains(MEGUMIN),
        eq(true)
    );
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn synapse_openai_path_anthropic_response_style() {
    let client = tensorzero::test_helpers::make_embedded_gateway_no_config().await;
    let mut headers = HeaderMap::new();
    headers.insert("x-synapse-response-style", "anthropic".parse().unwrap());
    let (status, _, body) = json_of(
        chat_completions_handler(
            State(client.get_app_state_data().unwrap().load_latest()),
            None,
            headers,
            chat_body("dummy::good"),
        )
        .await
        .unwrap(),
    )
    .await;
    assert_that!(status, eq(StatusCode::OK));
    expect_that!(body["type"].as_str().unwrap(), eq("message"));
    expect_that!(
        body["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains(MEGUMIN),
        eq(true)
    );
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn synapse_stream_aggregate_merges_content_deltas() {
    let client = tensorzero::test_helpers::make_embedded_gateway_no_config().await;
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-synapse-stream-aggregate",
        r#"[{"part":"content","startDelayMs":0,"intervalMs":10000,"maxChars":5000}]"#
            .parse()
            .unwrap(),
    );
    let (status, _, body) = text_of(
        chat_completions_handler(
            State(client.get_app_state_data().unwrap().load_latest()),
            None,
            headers,
            OpenAIStructuredJson(
                serde_json::from_value(json!({
                    "model": "dummy::good",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": true,
                }))
                .unwrap(),
            ),
        )
        .await
        .unwrap(),
    )
    .await;
    assert_that!(status, eq(StatusCode::OK));
    let events = sse_data_values(&body);
    let content_deltas: Vec<_> = events
        .iter()
        .filter_map(|event| event.pointer("/choices/0/delta/content"))
        .filter(|value| value.as_str().is_some_and(|text| !text.is_empty()))
        .collect();
    assert_that!(content_deltas.len(), eq(1));
    expect_that!(
        content_deltas[0].as_str().unwrap().contains(WALLY),
        eq(true)
    );
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn synapse_long_audio_eval_profile_and_unknown_profile() {
    let client = tensorzero::test_helpers::make_embedded_gateway_no_config().await;
    let mut ok_headers = HeaderMap::new();
    ok_headers.insert(
        "x-synapse-request-profile",
        "long-audio-eval".parse().unwrap(),
    );
    let (status, _, body) = json_of(
        chat_completions_handler(
            State(client.get_app_state_data().unwrap().load_latest()),
            None,
            ok_headers,
            chat_body("dummy::good"),
        )
        .await
        .unwrap(),
    )
    .await;
    assert_that!(status, eq(StatusCode::OK));
    expect_that!(
        body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .contains(MEGUMIN),
        eq(true)
    );
    expect_that!(LONG_AUDIO_EVAL_TIMEOUT, eq(Duration::from_secs(25 * 60)));

    let mut bad_headers = HeaderMap::new();
    bad_headers.insert(
        "x-synapse-request-profile",
        "not-a-profile".parse().unwrap(),
    );
    let (status, _, _) = json_of(
        chat_completions_handler(
            State(client.get_app_state_data().unwrap().load_latest()),
            None,
            bad_headers,
            chat_body("dummy::good"),
        )
        .await
        .unwrap(),
    )
    .await;
    assert_that!(status, eq(StatusCode::BAD_REQUEST));

    let mut empty_agg = HeaderMap::new();
    empty_agg.insert("x-synapse-stream-aggregate", "[]".parse().unwrap());
    let (status, _, _) = json_of(
        chat_completions_handler(
            State(client.get_app_state_data().unwrap().load_latest()),
            None,
            empty_agg,
            chat_body("dummy::good"),
        )
        .await
        .unwrap(),
    )
    .await;
    assert_that!(status, eq(StatusCode::BAD_REQUEST));
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn synapse_live_anthropic_messages_path() {
    let Some(http) = live_gateway().await else {
        println!("skip synapse_live_anthropic_messages_path: gateway not reachable");
        return;
    };
    let response = http
        .post(get_gateway_endpoint("/v1/messages"))
        .json(&json!({
            "model": "dummy::good",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Hi"}],
        }))
        .send()
        .await
        .unwrap();
    assert_that!(response.status(), eq(StatusCode::OK));
    let body: Value = response.json().await.unwrap();
    expect_that!(body["type"].as_str().unwrap(), eq("message"));
}

// ---------------------------------------------------------------------------
// D — cache + keys
// ---------------------------------------------------------------------------

#[gtest]
fn synapse_default_cache_on_unless_header_disables() {
    let on = resolve_cache_options(None, false);
    expect_that!(on.enabled, eq(CacheEnabledMode::On));
    let off = resolve_cache_options(None, true);
    expect_that!(off.enabled, eq(CacheEnabledMode::Off));
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn synapse_cache_false_header_still_serves() {
    let client = tensorzero::test_helpers::make_embedded_gateway_no_config().await;
    let mut headers = HeaderMap::new();
    headers.insert("x-synapse-cache", "false".parse().unwrap());
    let (status, _, body) = json_of(
        chat_completions_handler(
            State(client.get_app_state_data().unwrap().load_latest()),
            None,
            headers,
            chat_body("dummy::good"),
        )
        .await
        .unwrap(),
    )
    .await;
    assert_that!(status, eq(StatusCode::OK));
    expect_that!(
        body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .contains(MEGUMIN),
        eq(true)
    );
}

#[gtest]
fn synapse_api_key_format() {
    let rest: String = "a".repeat(48);
    let key = format!("sk-syn-v1-{rest}");
    expect_that!(key.starts_with("sk-syn-v1-"), eq(true));
    expect_that!(rest.len(), eq(48));
    expect_that!(
        rest.chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit()),
        eq(true)
    );
    expect_that!(
        "sk-t0-123456789012-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            .starts_with("sk-syn-v1-"),
        eq(false)
    );
}

// ---------------------------------------------------------------------------
// E — observability
// ---------------------------------------------------------------------------

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn synapse_embeddings_dummy() {
    let config = r#"
[embedding_models.qwen3-embedding-4b]
routing = ["dummy"]

[embedding_models.qwen3-embedding-4b.providers.dummy]
type = "dummy"
model_name = "test-embeddings"
"#;
    let client = tensorzero::test_helpers::make_embedded_gateway_with_config(config).await;
    let marker = format!("obs-embed-{}", Uuid::now_v7());
    let (status, _, body) = json_of(
        embeddings_handler(
            State(client.get_app_state_data().unwrap().load_latest()),
            None,
            HeaderMap::new(),
            OpenAIStructuredJson(
                serde_json::from_value(json!({
                    "model": "qwen3-embedding-4b",
                    "input": [marker, "world"],
                }))
                .unwrap(),
            ),
        )
        .await
        .unwrap(),
    )
    .await;
    assert_that!(status, eq(StatusCode::OK));
    expect_that!(body["object"].as_str().unwrap(), eq("list"));
    expect_that!(body["data"].as_array().unwrap().len(), eq(2));
    expect_that!(
        body["data"][0]["embedding"].as_array().unwrap().len(),
        eq(1536)
    );
    assert_standalone_inference_recorded(
        &marker,
        "qwen3-embedding-4b",
        "embeddings",
        "Generated 2 embeddings",
    )
    .await;
}

async fn assert_standalone_inference_recorded(
    marker: &str,
    variant_name: &str,
    endpoint: &str,
    output_contains: &str,
) {
    let function_name = match endpoint {
        "embeddings" => "tensorzero::embedding",
        "rerank" => "tensorzero::rerank",
        _ => "tensorzero::default",
    };
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let config = get_e2e_config().await;
    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                function_name: Some(function_name),
                variant_name: Some(variant_name),
                limit: 50,
                ..Default::default()
            },
        )
        .await
        .expect("list inferences");
    let chat = inferences
        .iter()
        .find_map(|inference| match inference {
            StoredInferenceDatabase::Chat(chat)
                if serde_json::to_string(&chat.input)
                    .unwrap_or_default()
                    .contains(marker) =>
            {
                Some(chat)
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("expected chat inference containing {marker}"));
    expect_that!(chat.function_name.as_str(), eq(function_name));
    expect_that!(
        chat.tags.get("tensorzero::endpoint").map(String::as_str),
        eq(Some(endpoint))
    );
    let output = serde_json::to_string(&chat.output).unwrap_or_default();
    expect_that!(output.contains(output_contains), eq(true));
    expect_that!(output.contains("0.1,"), eq(false));
    let model_inferences = conn
        .get_model_inferences_by_inference_id(chat.inference_id)
        .await
        .expect("model inferences");
    expect_that!(model_inferences.len(), eq(1));
    let raw_request = model_inferences[0]
        .raw_request
        .as_deref()
        .expect("model inference raw_request");
    expect_that!(raw_request, not(eq("")));
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn synapse_usage_analytics_and_balances() {
    let client = tensorzero::test_helpers::make_embedded_gateway_no_config().await;
    let state = client.get_app_state_data().unwrap().load_latest();
    let from = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
    let to = Utc.with_ymd_and_hms(2026, 1, 8, 0, 0, 0).unwrap();
    let csv = usage_export_handler(
        State(state.clone()),
        Query(SynapseTimeRangeQuery {
            from,
            to,
            tags: None,
            group_by_tag: None,
        }),
    )
    .await;
    let analytics = analytics_handler(
        State(state.clone()),
        Query(SynapseTimeRangeQuery {
            from,
            to,
            tags: None,
            group_by_tag: None,
        }),
    )
    .await;
    let balances = balances_handler(State(state)).await;

    match csv {
        Ok(response) => {
            expect_that!(
                response
                    .headers()
                    .get(axum::http::header::CONTENT_TYPE)
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .contains("text/csv"),
                eq(true)
            );
        }
        Err(error) => {
            expect_that!(
                error.to_string().to_ascii_lowercase().contains("postgres"),
                eq(true)
            );
        }
    }

    match analytics {
        Ok(json) => {
            let _rows = json.0.data;
        }
        Err(error) => {
            expect_that!(
                error.to_string().to_ascii_lowercase().contains("postgres"),
                eq(true)
            );
        }
    }

    let deepseek_key = std::env::var("DEEPSEEK_API_KEY")
        .ok()
        .filter(|k| !k.is_empty());
    let openrouter_key = std::env::var("OPENROUTER_API_KEY")
        .ok()
        .filter(|k| !k.is_empty());
    if deepseek_key.is_none() && openrouter_key.is_none() {
        let json = balances.expect("balances with no vendor keys must succeed");
        expect_that!(json.0.deepseek.is_none(), eq(true));
        expect_that!(json.0.openrouter.is_none(), eq(true));
    } else {
        println!("skip asserting mocked balances: vendor keys are set in the environment");
    }
}

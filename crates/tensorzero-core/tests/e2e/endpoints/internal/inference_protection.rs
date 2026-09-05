// Modified by Delta-AI under Apache 2.0
//! E2E tests for the internal inference protection endpoints.

use std::collections::HashMap;

use googletest::prelude::*;
use reqwest::Client;
use reqwest::StatusCode;
use uuid::Uuid;

use tensorzero_core::db::inferences::InferenceQueries;
use tensorzero_core::endpoints::inference::InferenceParams;
use tensorzero_core::endpoints::internal::inference_protection::{
    GetInferencesProtectionRequest, InferenceProtectionResponse, InferencesProtectionResponse,
    SetInferenceProtectionRequest,
};
use tensorzero_core::inference::types::ChatInferenceDatabaseInsert;
use tensorzero_core::inference::types::extra_body::UnfilteredInferenceExtraBody;
use tensorzero_core::inference::types::stored_input::StoredInput;
use tensorzero_core::tool::ToolCallConfigDatabaseInsert;

use crate::common::get_gateway_endpoint;
use crate::db::get_test_postgres;

fn make_chat_inference(function_name: &str) -> ChatInferenceDatabaseInsert {
    ChatInferenceDatabaseInsert {
        id: Uuid::now_v7(),
        function_name: function_name.to_string(),
        variant_name: "test_variant".to_string(),
        episode_id: Uuid::now_v7(),
        input: Some(StoredInput::default()),
        output: Some(vec![]),
        tool_params: Some(ToolCallConfigDatabaseInsert::default()),
        inference_params: Some(InferenceParams::default()),
        processing_time_ms: Some(42),
        ttft_ms: None,
        tags: HashMap::new(),
        extra_body: Some(UnfilteredInferenceExtraBody::default()),
        snapshot_hash: None,
    }
}

/// Full HTTP roundtrip: protect an inference, read its protection state,
/// unprotect it, and confirm the state is gone.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_inference_protection_http_roundtrip() {
    let conn = get_test_postgres().await;
    let function_name = format!("test_protection_endpoint_{}", Uuid::now_v7());
    let inference = make_chat_inference(&function_name);
    let inference_id = inference.id;
    conn.insert_chat_inferences(std::slice::from_ref(&inference))
        .await
        .expect("direct insert should succeed");

    let client = Client::new();

    // Protect
    let resp = client
        .post(get_gateway_endpoint(&format!(
            "/internal/inferences/{inference_id}/protection"
        )))
        .json(&SetInferenceProtectionRequest { protected: true })
        .send()
        .await
        .expect("set protection request should send");
    assert_that!(
        resp.status(),
        eq(StatusCode::OK),
        "set protection should succeed"
    );
    let body: InferenceProtectionResponse = resp.json().await.expect("response should parse");
    expect_that!(body.inference_id, eq(inference_id));
    expect_that!(body.function_type.as_str(), eq("chat"));
    expect_that!(body.protected_at, some(anything()));

    // Read back
    let resp = client
        .post(get_gateway_endpoint("/internal/inferences/protection"))
        .json(&GetInferencesProtectionRequest {
            ids: vec![inference_id],
        })
        .send()
        .await
        .expect("get protection request should send");
    assert_that!(resp.status(), eq(StatusCode::OK));
    let body: InferencesProtectionResponse = resp.json().await.expect("response should parse");
    assert_that!(
        body.protection.len(),
        eq(1),
        "protected inference should be returned"
    );
    expect_that!(body.protection[0].id, eq(inference_id));
    expect_that!(body.protection[0].function_type.as_str(), eq("chat"));

    // Unprotect
    let resp = client
        .post(get_gateway_endpoint(&format!(
            "/internal/inferences/{inference_id}/protection"
        )))
        .json(&SetInferenceProtectionRequest { protected: false })
        .send()
        .await
        .expect("unset protection request should send");
    assert_that!(resp.status(), eq(StatusCode::OK));
    let body: InferenceProtectionResponse = resp.json().await.expect("response should parse");
    expect_that!(body.protected_at, none());

    let resp = client
        .post(get_gateway_endpoint("/internal/inferences/protection"))
        .json(&GetInferencesProtectionRequest {
            ids: vec![inference_id],
        })
        .send()
        .await
        .expect("get protection request should send");
    let body: InferencesProtectionResponse = resp.json().await.expect("response should parse");
    expect_that!(
        body.protection.len(),
        eq(0),
        "unprotected inference should no longer be returned"
    );
}

/// Protecting an unknown inference id returns 404.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_set_protection_unknown_inference_returns_404() {
    let client = Client::new();
    let unknown_id = Uuid::now_v7();
    let resp = client
        .post(get_gateway_endpoint(&format!(
            "/internal/inferences/{unknown_id}/protection"
        )))
        .json(&SetInferenceProtectionRequest { protected: true })
        .send()
        .await
        .expect("request should send");
    expect_that!(
        resp.status(),
        eq(StatusCode::NOT_FOUND),
        "unknown inference id should return 404"
    );
}

/// The batch protection lookup accepts at most 1000 ids.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_get_protection_too_many_ids_returns_400() {
    let client = Client::new();
    let resp = client
        .post(get_gateway_endpoint("/internal/inferences/protection"))
        .json(&GetInferencesProtectionRequest {
            ids: (0..1001).map(|_| Uuid::now_v7()).collect(),
        })
        .send()
        .await
        .expect("request should send");
    expect_that!(
        resp.status(),
        eq(StatusCode::BAD_REQUEST),
        "more than 1000 ids should return 400"
    );
}

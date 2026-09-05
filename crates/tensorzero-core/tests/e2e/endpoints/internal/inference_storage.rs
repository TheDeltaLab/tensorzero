// Modified by Delta-AI under Apache 2.0
//! E2E tests for the internal inference storage stats / retention endpoints.

use googletest::prelude::*;
use reqwest::Client;
use reqwest::StatusCode;

use tensorzero_core::endpoints::internal::inference_storage::{
    InferenceRetentionConfig, InferenceStorageStatsResponse, UpdateInferenceRetentionRequest,
};

use crate::common::get_gateway_endpoint;
use crate::db::get_test_postgres;
use crate::db::postgres::{
    lock_retention_config, restore_retention_config, snapshot_retention_config,
};

/// `GET /internal/inference_storage/stats` returns stats for all 10
/// inference/archive tables.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_get_inference_storage_stats_endpoint() {
    let client = Client::new();
    let resp = client
        .get(get_gateway_endpoint("/internal/inference_storage/stats"))
        .send()
        .await
        .expect("request should send");
    assert_that!(
        resp.status(),
        eq(StatusCode::OK),
        "stats request should succeed"
    );

    let body: InferenceStorageStatsResponse = resp.json().await.expect("response should parse");
    expect_that!(
        body.tables.len(),
        eq(10),
        "stats should cover all 10 inference/archive tables"
    );
    for table in &body.tables {
        expect_that!(
            table.total_bytes,
            ge(0),
            "total_bytes should be non-negative for `{}`",
            table.name
        );
    }
}

/// Retention values must be >= 1 when present; 0 is rejected with 400.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_update_retention_rejects_zero() {
    let client = Client::new();
    for request in [
        UpdateInferenceRetentionRequest {
            metadata_retention_days: Some(0),
            data_retention_days: None,
        },
        UpdateInferenceRetentionRequest {
            metadata_retention_days: None,
            data_retention_days: Some(0),
        },
    ] {
        let resp = client
            .post(get_gateway_endpoint(
                "/internal/inference_storage/retention",
            ))
            .json(&request)
            .send()
            .await
            .expect("request should send");
        expect_that!(
            resp.status(),
            eq(StatusCode::BAD_REQUEST),
            "retention value of 0 should return 400"
        );
    }
}

/// Retention roundtrip over HTTP: set both values, then clear them (null =
/// keep forever). Restores the prior database state afterwards.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_update_retention_http_roundtrip() {
    let _guard = lock_retention_config().await;
    let conn = get_test_postgres().await;
    let pool = conn.get_pool().expect("Pool should be available").clone();
    let snapshot = snapshot_retention_config(&pool).await;

    let client = Client::new();
    let url = get_gateway_endpoint("/internal/inference_storage/retention");

    let resp = client
        .post(url.clone())
        .json(&UpdateInferenceRetentionRequest {
            metadata_retention_days: Some(30),
            data_retention_days: Some(14),
        })
        .send()
        .await
        .expect("request should send");
    assert_that!(resp.status(), eq(StatusCode::OK));
    let body: InferenceRetentionConfig = resp.json().await.expect("response should parse");
    expect_that!(body.metadata_retention_days, some(eq(30)));
    expect_that!(body.data_retention_days, some(eq(14)));

    // Absent values mean "keep forever" and delete the keys.
    let resp = client
        .post(url)
        .json(&UpdateInferenceRetentionRequest {
            metadata_retention_days: None,
            data_retention_days: None,
        })
        .send()
        .await
        .expect("request should send");
    assert_that!(resp.status(), eq(StatusCode::OK));
    let body: InferenceRetentionConfig = resp.json().await.expect("response should parse");
    expect_that!(body.metadata_retention_days, none());
    expect_that!(body.data_retention_days, none());

    restore_retention_config(&pool, &snapshot).await;
}

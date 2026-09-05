// Modified by Delta-AI under Apache 2.0
//! E2E tests for per-inference protection from retention cleanup, including the
//! `archive_and_drop_old_*_partitions` cleanup functions.

use std::collections::HashMap;

use googletest::prelude::*;
use uuid::Uuid;

use tensorzero_core::db::inferences::InferenceQueries;
use tensorzero_core::endpoints::inference::InferenceParams;
use tensorzero_core::error::ErrorDetails;
use tensorzero_core::inference::types::extra_body::UnfilteredInferenceExtraBody;
use tensorzero_core::inference::types::stored_input::StoredInput;
use tensorzero_core::inference::types::{
    ChatInferenceDatabaseInsert, JsonInferenceDatabaseInsert, JsonInferenceOutput,
};
use tensorzero_core::tool::ToolCallConfigDatabaseInsert;

use crate::db::get_test_postgres;
use crate::db::postgres::{
    lock_retention_config, restore_retention_config, snapshot_retention_config,
};

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

fn make_json_inference(function_name: &str) -> JsonInferenceDatabaseInsert {
    JsonInferenceDatabaseInsert {
        id: Uuid::now_v7(),
        function_name: function_name.to_string(),
        variant_name: "test_variant".to_string(),
        episode_id: Uuid::now_v7(),
        input: Some(StoredInput::default()),
        output: Some(JsonInferenceOutput {
            raw: Some("{}".to_string()),
            parsed: Some(serde_json::json!({})),
        }),
        auxiliary_content: Some(vec![]),
        inference_params: Some(InferenceParams::default()),
        processing_time_ms: Some(42),
        output_schema: Some(serde_json::json!({})),
        ttft_ms: None,
        tags: HashMap::new(),
        extra_body: Some(UnfilteredInferenceExtraBody::default()),
        snapshot_hash: None,
    }
}

/// Protecting and unprotecting a live chat inference toggles its entry in
/// `get_inferences_protection`.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_chat_inference_protection_roundtrip() {
    let conn = get_test_postgres().await;
    let function_name = format!("test_protection_chat_{}", Uuid::now_v7());
    let inference = make_chat_inference(&function_name);
    let inference_id = inference.id;
    conn.insert_chat_inferences(std::slice::from_ref(&inference))
        .await
        .expect("direct insert should succeed");

    // Protect
    let (function_type, protected_at) = conn
        .set_inference_protection(inference_id, true)
        .await
        .expect("protecting a live chat inference should succeed");
    expect_that!(function_type.as_str(), eq("chat"));
    expect_that!(
        protected_at,
        some(anything()),
        "protecting should set `protected_at`"
    );

    // A protected id is returned; an unrelated unprotected id is not.
    let other_id = Uuid::now_v7();
    let protection = conn
        .get_inferences_protection(&[inference_id, other_id])
        .await
        .expect("get_inferences_protection should succeed");
    assert_that!(
        protection.len(),
        eq(1),
        "only the protected inference should be returned"
    );
    let row = &protection[0];
    expect_that!(
        row.id,
        eq(inference_id),
        "returned row should be the protected inference"
    );
    expect_that!(
        row.function_type.as_str(),
        eq("chat"),
        "function_type should be `chat`"
    );

    // Unprotect
    let (function_type, protected_at) = conn
        .set_inference_protection(inference_id, false)
        .await
        .expect("unprotecting a live chat inference should succeed");
    expect_that!(function_type.as_str(), eq("chat"));
    expect_that!(
        protected_at,
        none(),
        "unprotecting should clear `protected_at`"
    );

    let protection = conn
        .get_inferences_protection(&[inference_id])
        .await
        .expect("get_inferences_protection should succeed");
    expect_that!(
        protection.len(),
        eq(0),
        "unprotected inference should no longer be returned"
    );
}

/// Same roundtrip for a JSON inference.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_json_inference_protection_roundtrip() {
    let conn = get_test_postgres().await;
    let function_name = format!("test_protection_json_{}", Uuid::now_v7());
    let inference = make_json_inference(&function_name);
    let inference_id = inference.id;
    conn.insert_json_inferences(std::slice::from_ref(&inference))
        .await
        .expect("direct insert should succeed");

    let (function_type, protected_at) = conn
        .set_inference_protection(inference_id, true)
        .await
        .expect("protecting a live json inference should succeed");
    expect_that!(function_type.as_str(), eq("json"));
    expect_that!(protected_at, some(anything()));

    let protection = conn
        .get_inferences_protection(&[inference_id])
        .await
        .expect("get_inferences_protection should succeed");
    assert_that!(protection.len(), eq(1));
    let row = &protection[0];
    expect_that!(
        row.id,
        eq(inference_id),
        "returned row should be the protected inference"
    );
    expect_that!(
        row.function_type.as_str(),
        eq("json"),
        "function_type should be `json`"
    );

    let (_, protected_at) = conn
        .set_inference_protection(inference_id, false)
        .await
        .expect("unprotecting a live json inference should succeed");
    expect_that!(protected_at, none());

    let protection = conn
        .get_inferences_protection(&[inference_id])
        .await
        .expect("get_inferences_protection should succeed");
    expect_that!(protection.len(), eq(0));
}

/// Setting protection on an id that exists in neither the live nor the archive
/// tables returns an `InferenceNotFound` error.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_set_protection_unknown_inference_errors() {
    let conn = get_test_postgres().await;
    let unknown_id = Uuid::now_v7();

    let err = conn
        .set_inference_protection(unknown_id, true)
        .await
        .expect_err("protecting an unknown inference id should fail");
    assert!(
        matches!(
            err.get_details(),
            ErrorDetails::InferenceNotFound { inference_id } if *inference_id == unknown_id
        ),
        "unknown id should produce an InferenceNotFound error, got: {err:?}"
    );
}

/// Archived inferences are already protected forever: re-protecting is a no-op
/// success, unprotecting is rejected, and the archive row is reported by
/// `get_inferences_protection`.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_archived_inference_protection() {
    let conn = get_test_postgres().await;
    let pool = conn.get_pool().expect("Pool should be available").clone();
    let inference_id = Uuid::now_v7();

    // Insert directly into the archive table, deriving `created_at` from the
    // UUIDv7 exactly like the live tables do.
    sqlx::query(
        "INSERT INTO tensorzero.chat_inferences_archive \
         (id, function_name, variant_name, episode_id, created_at, protected_at) \
         VALUES ($1, 'test_archived_protection', 'test_variant', $2, \
                 tensorzero.uuid_v7_to_timestamp($1::uuid), NOW())",
    )
    .bind(inference_id)
    .bind(Uuid::now_v7())
    .execute(&pool)
    .await
    .expect("inserting into `chat_inferences_archive` should succeed");

    let (function_type, protected_at) = conn
        .set_inference_protection(inference_id, true)
        .await
        .expect("protecting an archived inference should be a no-op success");
    expect_that!(function_type.as_str(), eq("chat"));
    expect_that!(
        protected_at,
        some(anything()),
        "archived inference should keep its existing `protected_at`"
    );

    let err = conn
        .set_inference_protection(inference_id, false)
        .await
        .expect_err("unprotecting an archived inference should fail");
    assert!(
        matches!(err.get_details(), ErrorDetails::InvalidRequest { .. }),
        "unprotecting an archived inference should produce an InvalidRequest error, got: {err:?}"
    );

    let protection = conn
        .get_inferences_protection(&[inference_id])
        .await
        .expect("get_inferences_protection should succeed");
    assert_that!(
        protection.len(),
        eq(1),
        "archived protected inference should be returned"
    );
    let row = &protection[0];
    expect_that!(
        row.id,
        eq(inference_id),
        "returned row should be the archived inference"
    );
    expect_that!(
        row.function_type.as_str(),
        eq("chat"),
        "function_type should be `chat`"
    );

    // Tidy up the archive table.
    sqlx::query("DELETE FROM tensorzero.chat_inferences_archive WHERE id = $1")
        .bind(inference_id)
        .execute(&pool)
        .await
        .expect("cleaning up archive row should succeed");
}

async fn partition_exists(pool: &sqlx::PgPool, table_name: &str) -> bool {
    sqlx::query_scalar(
        "SELECT EXISTS (SELECT 1 FROM pg_tables WHERE schemaname = 'tensorzero' AND tablename = $1)",
    )
    .bind(table_name)
    .fetch_one(pool)
    .await
    .expect("partition existence query should succeed")
}

async fn archive_row_count(pool: &sqlx::PgPool, table: &str, id: Uuid) -> i64 {
    let mut qb = sqlx::QueryBuilder::<sqlx::Postgres>::new("SELECT COUNT(*)::BIGINT FROM ");
    qb.push(table);
    qb.push(" WHERE id = ");
    qb.push_bind(id);
    qb.build_query_scalar()
        .fetch_one(pool)
        .await
        .expect("archive count query should succeed")
}

/// Drops one of this test's partitions (trusted, hardcoded table names only).
async fn drop_partition(pool: &sqlx::PgPool, partition: &str) {
    let mut qb = sqlx::QueryBuilder::<sqlx::Postgres>::new("DROP TABLE IF EXISTS tensorzero.");
    qb.push(partition);
    qb.build()
        .execute(pool)
        .await
        .expect("dropping test partition should succeed");
}

/// End-to-end test of the retention cleanup functions
/// (`archive_and_drop_old_metadata_partitions` /
/// `archive_and_drop_old_data_partitions`) on a manually-created old
/// monthly/daily partition pair:
/// - with retention unconfigured, the functions must not touch anything;
/// - with retention configured, the old partitions are dropped, protected rows
///   (metadata + payload) land in the archive tables, and unprotected rows are
///   dropped with their partition.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_archive_and_drop_old_partitions() {
    let _guard = lock_retention_config().await;
    let conn = get_test_postgres().await;
    let pool = conn.get_pool().expect("Pool should be available").clone();
    let snapshot = snapshot_retention_config(&pool).await;

    let protected_id = Uuid::now_v7();
    let unprotected_id = Uuid::now_v7();
    let metadata_partition = "chat_inferences_2025_01";
    let data_partition = "chat_inference_data_2025_01_15";

    // Idempotency across reruns: drop leftovers from an aborted previous run.
    for partition in [metadata_partition, data_partition] {
        drop_partition(&pool, partition).await;
    }

    sqlx::query(
        "CREATE TABLE tensorzero.chat_inferences_2025_01 \
         PARTITION OF tensorzero.chat_inferences \
         FOR VALUES FROM ('2025-01-01') TO ('2025-02-01')",
    )
    .execute(&pool)
    .await
    .expect("creating old monthly metadata partition should succeed");
    sqlx::query(
        "CREATE TABLE tensorzero.chat_inference_data_2025_01_15 \
         PARTITION OF tensorzero.chat_inference_data \
         FOR VALUES FROM ('2025-01-15') TO ('2025-01-16')",
    )
    .execute(&pool)
    .await
    .expect("creating old daily data partition should succeed");

    // One protected and one unprotected inference, both in the old partitions.
    sqlx::query(
        "INSERT INTO tensorzero.chat_inferences \
         (id, function_name, variant_name, episode_id, created_at, protected_at) \
         VALUES ($1, 'test_archive_protected', 'test_variant', $3, \
                 '2025-01-15 12:00:00+00', '2025-01-15 12:00:00+00'), \
                ($2, 'test_archive_unprotected', 'test_variant', $3, \
                 '2025-01-15 13:00:00+00', NULL)",
    )
    .bind(protected_id)
    .bind(unprotected_id)
    .bind(Uuid::now_v7())
    .execute(&pool)
    .await
    .expect("inserting old metadata rows should succeed");
    sqlx::query(
        "INSERT INTO tensorzero.chat_inference_data \
         (id, input, output, inference_params, created_at) \
         VALUES ($1, '{}', '[]', '{}', '2025-01-15 12:00:00+00'), \
                ($2, '{}', '[]', '{}', '2025-01-15 13:00:00+00')",
    )
    .bind(protected_id)
    .bind(unprotected_id)
    .execute(&pool)
    .await
    .expect("inserting old data rows should succeed");

    // Phase 1: retention not configured -> cleanup functions must no-op.
    conn.set_inference_retention(None, None)
        .await
        .expect("clearing retention config should succeed");
    sqlx::query("SELECT tensorzero.archive_and_drop_old_metadata_partitions()")
        .execute(&pool)
        .await
        .expect("metadata cleanup without retention configured should not error");
    sqlx::query("SELECT tensorzero.archive_and_drop_old_data_partitions()")
        .execute(&pool)
        .await
        .expect("data cleanup without retention configured should not error");
    expect_that!(
        partition_exists(&pool, metadata_partition).await,
        eq(true),
        "metadata partition must survive when retention is not configured"
    );
    expect_that!(
        partition_exists(&pool, data_partition).await,
        eq(true),
        "data partition must survive when retention is not configured"
    );

    // Phase 2: retention configured -> protected rows archived, partitions dropped.
    conn.set_inference_retention(Some(30), Some(30))
        .await
        .expect("setting retention config should succeed");
    sqlx::query("SELECT tensorzero.archive_and_drop_old_metadata_partitions()")
        .execute(&pool)
        .await
        .expect("metadata cleanup should succeed");
    sqlx::query("SELECT tensorzero.archive_and_drop_old_data_partitions()")
        .execute(&pool)
        .await
        .expect("data cleanup should succeed");

    expect_that!(
        partition_exists(&pool, metadata_partition).await,
        eq(false),
        "old metadata partition should be dropped"
    );
    expect_that!(
        partition_exists(&pool, data_partition).await,
        eq(false),
        "old data partition should be dropped"
    );

    // Protected rows landed in the archive tables.
    expect_that!(
        archive_row_count(&pool, "tensorzero.chat_inferences_archive", protected_id).await,
        eq(1),
        "protected metadata row should be archived"
    );
    expect_that!(
        archive_row_count(
            &pool,
            "tensorzero.chat_inference_data_archive",
            protected_id
        )
        .await,
        eq(1),
        "protected data row should be archived"
    );
    // Unprotected rows were dropped with their partitions, not archived.
    expect_that!(
        archive_row_count(&pool, "tensorzero.chat_inferences_archive", unprotected_id).await,
        eq(0),
        "unprotected metadata row should not be archived"
    );
    expect_that!(
        archive_row_count(
            &pool,
            "tensorzero.chat_inference_data_archive",
            unprotected_id
        )
        .await,
        eq(0),
        "unprotected data row should not be archived"
    );

    // The archived inference is still reported as protected.
    let protection = conn
        .get_inferences_protection(&[protected_id])
        .await
        .expect("get_inferences_protection should succeed");
    expect_that!(
        protection.len(),
        eq(1),
        "archived inference should still be reported as protected after cleanup"
    );

    // Tidy up: remove archived test rows and restore the retention config.
    sqlx::query("DELETE FROM tensorzero.chat_inferences_archive WHERE id = $1")
        .bind(protected_id)
        .execute(&pool)
        .await
        .expect("cleaning up archived metadata row should succeed");
    sqlx::query("DELETE FROM tensorzero.chat_inference_data_archive WHERE id = $1")
        .bind(protected_id)
        .execute(&pool)
        .await
        .expect("cleaning up archived data row should succeed");
    restore_retention_config(&pool, &snapshot).await;
}

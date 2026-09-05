// Modified by Delta-AI under Apache 2.0
//! E2E tests for Postgres inference storage stats and retention configuration.

use googletest::prelude::*;

use crate::db::get_test_postgres;
use crate::db::postgres::{
    lock_retention_config, restore_retention_config, snapshot_retention_config,
};

/// All inference/archive tables that `get_inference_storage_stats` must report.
const EXPECTED_TABLES: [&str; 10] = [
    "chat_inference_data",
    "chat_inference_data_archive",
    "chat_inferences",
    "chat_inferences_archive",
    "json_inference_data",
    "json_inference_data_archive",
    "json_inferences",
    "json_inferences_archive",
    "model_inference_data",
    "model_inferences",
];

/// Storage stats are catalog-only and must cover every inference table with
/// non-negative sizes and partition counts.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_get_inference_storage_stats_returns_all_tables() {
    let conn = get_test_postgres().await;

    let stats = conn
        .get_inference_storage_stats()
        .await
        .expect("get_inference_storage_stats should succeed");

    let mut names: Vec<&str> = stats.iter().map(|row| row.name.as_str()).collect();
    names.sort_unstable();
    expect_that!(
        names,
        container_eq(EXPECTED_TABLES),
        "storage stats should cover all 10 inference/archive tables"
    );

    for row in &stats {
        expect_that!(
            row.total_bytes,
            ge(0),
            "total_bytes should be non-negative for `{}`",
            row.name
        );
        expect_that!(
            row.partition_count,
            ge(0),
            "partition_count should be non-negative for `{}`",
            row.name
        );
        expect_that!(
            row.estimated_rows,
            ge(0),
            "estimated_rows should be non-negative for `{}`",
            row.name
        );
    }

    // The live inference tables are partitioned and the e2e database always has
    // at least the current-month/daily partitions created by migrations.
    let chat = stats
        .iter()
        .find(|row| row.name == "chat_inferences")
        .expect("stats should include `chat_inferences`");
    expect_that!(
        chat.partition_count,
        ge(1),
        "`chat_inferences` should have at least one partition"
    );
}

/// `set_inference_retention` upserts `Some` values and deletes keys for `None`
/// ("keep forever"); `get_inference_retention` reads them back.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_inference_retention_roundtrip() {
    let _guard = lock_retention_config().await;
    let conn = get_test_postgres().await;
    let pool = conn.get_pool().expect("Pool should be available").clone();
    let snapshot = snapshot_retention_config(&pool).await;

    conn.set_inference_retention(Some(30), None)
        .await
        .expect("set_inference_retention(Some(30), None) should succeed");
    let retention = conn
        .get_inference_retention()
        .await
        .expect("get_inference_retention should succeed");
    expect_that!(
        retention.metadata_retention_days,
        some(eq(30)),
        "metadata retention should be 30 days"
    );
    expect_that!(
        retention.data_retention_days,
        none(),
        "data retention key should be deleted"
    );

    // Roundtrip back: clear metadata, set data.
    conn.set_inference_retention(None, Some(14))
        .await
        .expect("set_inference_retention(None, Some(14)) should succeed");
    let retention = conn
        .get_inference_retention()
        .await
        .expect("get_inference_retention should succeed");
    expect_that!(
        retention.metadata_retention_days,
        none(),
        "metadata retention key should be deleted"
    );
    expect_that!(
        retention.data_retention_days,
        some(eq(14)),
        "data retention should be 14 days"
    );

    restore_retention_config(&pool, &snapshot).await;
}

/// `write_retention_config(None, None)` (the TOML-absent case) must leave
/// previously-set database keys intact, so dashboard-set values survive
/// gateway restarts.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_write_retention_config_none_preserves_db_keys() {
    let _guard = lock_retention_config().await;
    let conn = get_test_postgres().await;
    let pool = conn.get_pool().expect("Pool should be available").clone();
    let snapshot = snapshot_retention_config(&pool).await;

    conn.set_inference_retention(Some(30), Some(60))
        .await
        .expect("set_inference_retention should succeed");

    conn.write_retention_config(None, None)
        .await
        .expect("write_retention_config(None, None) should succeed");

    let retention = conn
        .get_inference_retention()
        .await
        .expect("get_inference_retention should succeed");
    expect_that!(
        retention.metadata_retention_days,
        some(eq(30)),
        "TOML-absent metadata retention must not delete the database key"
    );
    expect_that!(
        retention.data_retention_days,
        some(eq(60)),
        "TOML-absent data retention must not delete the database key"
    );

    restore_retention_config(&pool, &snapshot).await;
}

/// `write_retention_config(Some, Some)` overwrites the database values, and
/// always deletes the legacy `inference_retention_days` key.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_write_retention_config_overwrites_and_deletes_legacy_key() {
    let _guard = lock_retention_config().await;
    let conn = get_test_postgres().await;
    let pool = conn.get_pool().expect("Pool should be available").clone();
    let snapshot = snapshot_retention_config(&pool).await;

    // Seed a legacy key to be deleted.
    sqlx::query("INSERT INTO tensorzero.retention_config (key, value) VALUES ('inference_retention_days', '999') ON CONFLICT (key) DO UPDATE SET value = '999'")
        .execute(&pool)
        .await
        .expect("seeding legacy retention key should succeed");

    conn.write_retention_config(Some(7), Some(14))
        .await
        .expect("write_retention_config(Some(7), Some(14)) should succeed");

    let retention = conn
        .get_inference_retention()
        .await
        .expect("get_inference_retention should succeed");
    expect_that!(
        retention.metadata_retention_days,
        some(eq(7)),
        "TOML-set metadata retention should overwrite the database value"
    );
    expect_that!(
        retention.data_retention_days,
        some(eq(14)),
        "TOML-set data retention should overwrite the database value"
    );

    let legacy_count: i64 =
        sqlx::query_scalar("SELECT COUNT(*)::BIGINT FROM tensorzero.retention_config WHERE key = 'inference_retention_days'")
            .fetch_one(&pool)
            .await
            .expect("legacy key count query should succeed");
    expect_that!(
        legacy_count,
        eq(0),
        "legacy `inference_retention_days` key should always be deleted"
    );

    restore_retention_config(&pool, &snapshot).await;
}

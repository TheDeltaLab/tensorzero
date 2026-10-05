// Modified by Delta-AI under Apache 2.0
//! E2E tests for tag-based scheduled cleanup of Postgres payload tables.
//!
//! These tests drive `run_cleanup_pass` directly against the shared e2e
//! Postgres database: they create a cleanup rule, insert inference rows with
//! unique tag keys, run one manual pass, and assert on the surviving/deleted
//! rows and on the `cleanup_runs` / `cleanup_run_steps` bookkeeping.
//!
//! `run_cleanup_pass` applies *every* enabled rule in the database, so all
//! tests in this file serialize on a Postgres advisory lock (nextest runs each
//! test in its own process) — otherwise one test's pass could delete another
//! test's freshly-inserted rows before that test runs its own pass.

use chrono::{DateTime, Utc};
use googletest::prelude::*;
use sqlx::Connection as _;
use sqlx::PgPool;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use tensorzero_core::cleanup::run_cleanup_pass;
use tensorzero_core::db::postgres::cleanup as cleanup_db;

use crate::db::get_test_postgres;

/// Arbitrary `pg_advisory_lock` key serializing the tests in this file
/// against each other (see the module docs). Distinct from
/// `RETENTION_CONFIG_LOCK_KEY` in `mod.rs` so cleanup tests do not queue
/// behind retention tests.
const CLEANUP_LOCK_KEY: i64 = 861_033;

/// Holds a session-level Postgres advisory lock serializing cleanup tests.
/// The lock is released when the guard is dropped (dropping the connection
/// closes the session).
struct CleanupLock {
    _conn: sqlx::postgres::PgConnection,
}

async fn lock_cleanup() -> CleanupLock {
    let postgres_url = std::env::var("TENSORZERO_POSTGRES_URL")
        .expect("Environment variable TENSORZERO_POSTGRES_URL must be set");
    let mut conn = sqlx::postgres::PgConnection::connect(&postgres_url)
        .await
        .expect("connecting for the cleanup advisory lock should succeed");
    sqlx::query("SELECT pg_advisory_lock($1)")
        .bind(CLEANUP_LOCK_KEY)
        .execute(&mut conn)
        .await
        .expect("acquiring the cleanup advisory lock should succeed");
    CleanupLock { _conn: conn }
}

/// Ten days ago: old enough for the `older_than_days = 7` rules below, and old
/// enough to land outside today's active daily partition.
fn old_timestamp() -> DateTime<Utc> {
    Utc::now() - chrono::Duration::days(10)
}

/// A tag key unique to one test invocation, so tests never match each other's
/// rows (or fixture rows).
fn unique_tag_key() -> String {
    format!("test_cleanup_{}", Uuid::now_v7().simple())
}

async fn insert_chat_inference(
    pool: &PgPool,
    id: Uuid,
    tags: serde_json::Value,
    created_at: DateTime<Utc>,
    protected_at: Option<DateTime<Utc>>,
) {
    sqlx::query(
        "INSERT INTO tensorzero.chat_inferences \
         (id, function_name, variant_name, episode_id, tags, created_at, protected_at) \
         VALUES ($1, 'test_cleanup_fn', 'test_variant', $2, $3, $4, $5)",
    )
    .bind(id)
    .bind(Uuid::now_v7())
    .bind(tags)
    .bind(created_at)
    .bind(protected_at)
    .execute(pool)
    .await
    .expect("inserting a `chat_inferences` row should succeed");
}

async fn insert_chat_inference_data(pool: &PgPool, id: Uuid, created_at: DateTime<Utc>) {
    sqlx::query(
        "INSERT INTO tensorzero.chat_inference_data \
         (id, input, output, inference_params, created_at) \
         VALUES ($1, '{}', '[]', '{}', $2)",
    )
    .bind(id)
    .bind(created_at)
    .execute(pool)
    .await
    .expect("inserting a `chat_inference_data` row should succeed");
}

async fn insert_model_inference(
    pool: &PgPool,
    id: Uuid,
    inference_id: Uuid,
    created_at: DateTime<Utc>,
) {
    sqlx::query(
        "INSERT INTO tensorzero.model_inferences \
         (id, inference_id, model_name, model_provider_name, created_at) \
         VALUES ($1, $2, 'test_model', 'test_provider', $3)",
    )
    .bind(id)
    .bind(inference_id)
    .bind(created_at)
    .execute(pool)
    .await
    .expect("inserting a `model_inferences` row should succeed");
}

async fn insert_model_inference_data(pool: &PgPool, id: Uuid, created_at: DateTime<Utc>) {
    sqlx::query(
        "INSERT INTO tensorzero.model_inference_data \
         (id, raw_request, raw_response, input_messages, created_at) \
         VALUES ($1, '{}', '{}', '[]', $2)",
    )
    .bind(id)
    .bind(created_at)
    .execute(pool)
    .await
    .expect("inserting a `model_inference_data` row should succeed");
}

/// Counts rows matching `id` in one of the inference tables. Only called with
/// hardcoded table/column names from this file, so the identifiers are safe to
/// interpolate.
async fn count_rows(pool: &PgPool, table: &str, column: &str, id: Uuid) -> i64 {
    let mut qb = sqlx::QueryBuilder::<sqlx::Postgres>::new("SELECT COUNT(*)::BIGINT FROM ");
    qb.push(table);
    qb.push(" WHERE ");
    qb.push(column);
    qb.push(" = ");
    qb.push_bind(id);
    qb.build_query_scalar()
        .fetch_one(pool)
        .await
        .expect("row count query should succeed")
}

async fn count_chat_data(pool: &PgPool, id: Uuid) -> i64 {
    count_rows(pool, "tensorzero.chat_inference_data", "id", id).await
}

async fn count_chat_metadata(pool: &PgPool, id: Uuid) -> i64 {
    count_rows(pool, "tensorzero.chat_inferences", "id", id).await
}

async fn count_model_data(pool: &PgPool, id: Uuid) -> i64 {
    count_rows(pool, "tensorzero.model_inference_data", "id", id).await
}

/// Removes everything a test created: the rule, this test's run (steps cascade
/// via `ON DELETE CASCADE`), the tagged metadata rows, and the model inference
/// rows. Payload rows are already gone — deleting them was the point of the
/// test.
async fn tidy_up(
    pool: &PgPool,
    rule_id: Uuid,
    run_id: Uuid,
    tag_key: &str,
    model_inference_ids: &[Uuid],
) {
    cleanup_db::delete_cleanup_rule(pool, rule_id)
        .await
        .expect("deleting the cleanup rule should succeed");
    sqlx::query("DELETE FROM tensorzero.cleanup_runs WHERE id = $1")
        .bind(run_id)
        .execute(pool)
        .await
        .expect("deleting the cleanup run should succeed");
    sqlx::query("DELETE FROM tensorzero.chat_inferences WHERE tags ? $1")
        .bind(tag_key)
        .execute(pool)
        .await
        .expect("deleting tagged metadata rows should succeed");
    if !model_inference_ids.is_empty() {
        sqlx::query("DELETE FROM tensorzero.model_inferences WHERE id = ANY($1)")
            .bind(model_inference_ids)
            .execute(pool)
            .await
            .expect("deleting model inference rows should succeed");
    }
}

/// Fetches the steps of one (run, rule) pair.
async fn steps_for_rule(
    pool: &PgPool,
    run_id: Uuid,
    rule_id: Uuid,
) -> Vec<cleanup_db::CleanupRunStepRow> {
    cleanup_db::list_cleanup_run_steps(pool, &[run_id])
        .await
        .expect("listing cleanup run steps should succeed")
        .into_iter()
        .filter(|step| step.rule_id == Some(rule_id))
        .collect()
}

/// A rule matching `tag_key = 'yes'` and rows older than 7 days deletes old
/// tagged payload rows, leaves untagged payload rows alone, and never touches
/// the metadata table.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_cleanup_basic_match_deletes_payload_keeps_metadata() {
    let _guard = lock_cleanup().await;
    let conn = get_test_postgres().await;
    let pool = conn.get_pool().expect("Pool should be available").clone();

    let tag_key = unique_tag_key();
    let rule = cleanup_db::create_cleanup_rule(&pool, &tag_key, Some("yes"), 7, true)
        .await
        .expect("creating a cleanup rule should succeed");

    let tagged_id = Uuid::now_v7();
    let untagged_id = Uuid::now_v7();
    insert_chat_inference(
        &pool,
        tagged_id,
        serde_json::json!({&tag_key: "yes"}),
        old_timestamp(),
        None,
    )
    .await;
    insert_chat_inference_data(&pool, tagged_id, old_timestamp()).await;
    insert_chat_inference(
        &pool,
        untagged_id,
        serde_json::json!({}),
        old_timestamp(),
        None,
    )
    .await;
    insert_chat_inference_data(&pool, untagged_id, old_timestamp()).await;

    let run_id = run_cleanup_pass(
        &pool,
        cleanup_db::CLEANUP_TRIGGER_MANUAL,
        &CancellationToken::new(),
    )
    .await
    .expect("cleanup pass should succeed");

    expect_that!(
        count_chat_data(&pool, tagged_id).await,
        eq(0),
        "tagged payload row should be deleted"
    );
    expect_that!(
        count_chat_data(&pool, untagged_id).await,
        eq(1),
        "untagged payload row should survive"
    );
    expect_that!(
        count_chat_metadata(&pool, tagged_id).await,
        eq(1),
        "metadata rows are never deleted by tag-based cleanup"
    );

    tidy_up(&pool, rule.id, run_id, &tag_key, &[]).await;
}

/// A tagged old inference with `protected_at` set is exempt from cleanup.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_cleanup_skips_protected_inferences() {
    let _guard = lock_cleanup().await;
    let conn = get_test_postgres().await;
    let pool = conn.get_pool().expect("Pool should be available").clone();

    let tag_key = unique_tag_key();
    let rule = cleanup_db::create_cleanup_rule(&pool, &tag_key, Some("yes"), 7, true)
        .await
        .expect("creating a cleanup rule should succeed");

    let protected_id = Uuid::now_v7();
    insert_chat_inference(
        &pool,
        protected_id,
        serde_json::json!({&tag_key: "yes"}),
        old_timestamp(),
        Some(old_timestamp()),
    )
    .await;
    insert_chat_inference_data(&pool, protected_id, old_timestamp()).await;

    let run_id = run_cleanup_pass(
        &pool,
        cleanup_db::CLEANUP_TRIGGER_MANUAL,
        &CancellationToken::new(),
    )
    .await
    .expect("cleanup pass should succeed");

    expect_that!(
        count_chat_data(&pool, protected_id).await,
        eq(1),
        "payload of a protected inference must survive cleanup"
    );

    tidy_up(&pool, rule.id, run_id, &tag_key, &[]).await;
}

/// A matching row in today's active daily partition is never deleted, even
/// though it carries the tag.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_cleanup_skips_active_partition() {
    let _guard = lock_cleanup().await;
    let conn = get_test_postgres().await;
    let pool = conn.get_pool().expect("Pool should be available").clone();

    let tag_key = unique_tag_key();
    let rule = cleanup_db::create_cleanup_rule(&pool, &tag_key, Some("yes"), 7, true)
        .await
        .expect("creating a cleanup rule should succeed");

    let active_id = Uuid::now_v7();
    let now = Utc::now();
    insert_chat_inference(
        &pool,
        active_id,
        serde_json::json!({&tag_key: "yes"}),
        now,
        None,
    )
    .await;
    insert_chat_inference_data(&pool, active_id, now).await;

    let run_id = run_cleanup_pass(
        &pool,
        cleanup_db::CLEANUP_TRIGGER_MANUAL,
        &CancellationToken::new(),
    )
    .await
    .expect("cleanup pass should succeed");

    expect_that!(
        count_chat_data(&pool, active_id).await,
        eq(1),
        "rows in the current day's active partition must survive cleanup"
    );

    tidy_up(&pool, rule.id, run_id, &tag_key, &[]).await;
}

/// A rule with `tag_value = NULL` matches every value of the tag key (but not
/// rows lacking the key).
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_cleanup_null_tag_value_matches_any_value() {
    let _guard = lock_cleanup().await;
    let conn = get_test_postgres().await;
    let pool = conn.get_pool().expect("Pool should be available").clone();

    let tag_key = unique_tag_key();
    let rule = cleanup_db::create_cleanup_rule(&pool, &tag_key, None, 7, true)
        .await
        .expect("creating a cleanup rule should succeed");

    let value_a_id = Uuid::now_v7();
    let value_b_id = Uuid::now_v7();
    let no_key_id = Uuid::now_v7();
    for (id, tags) in [
        (value_a_id, serde_json::json!({&tag_key: "a"})),
        (value_b_id, serde_json::json!({&tag_key: "b"})),
        (no_key_id, serde_json::json!({})),
    ] {
        insert_chat_inference(&pool, id, tags, old_timestamp(), None).await;
        insert_chat_inference_data(&pool, id, old_timestamp()).await;
    }

    let run_id = run_cleanup_pass(
        &pool,
        cleanup_db::CLEANUP_TRIGGER_MANUAL,
        &CancellationToken::new(),
    )
    .await
    .expect("cleanup pass should succeed");

    expect_that!(
        count_chat_data(&pool, value_a_id).await,
        eq(0),
        "any value of the key should match a NULL-value rule"
    );
    expect_that!(
        count_chat_data(&pool, value_b_id).await,
        eq(0),
        "any value of the key should match a NULL-value rule"
    );
    expect_that!(
        count_chat_data(&pool, no_key_id).await,
        eq(1),
        "rows lacking the tag key should survive"
    );

    tidy_up(&pool, rule.id, run_id, &tag_key, &[]).await;
}

/// `model_inference_data` is cleaned through the chain chat inference (tags)
/// -> `model_inferences` (`inference_id`) -> `model_inference_data`.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_cleanup_model_inference_data_chain() {
    let _guard = lock_cleanup().await;
    let conn = get_test_postgres().await;
    let pool = conn.get_pool().expect("Pool should be available").clone();

    let tag_key = unique_tag_key();
    let rule = cleanup_db::create_cleanup_rule(&pool, &tag_key, Some("yes"), 7, true)
        .await
        .expect("creating a cleanup rule should succeed");

    // Tagged chat inference -> model inference -> model inference data.
    let tagged_inference_id = Uuid::now_v7();
    let tagged_model_id = Uuid::now_v7();
    insert_chat_inference(
        &pool,
        tagged_inference_id,
        serde_json::json!({&tag_key: "yes"}),
        old_timestamp(),
        None,
    )
    .await;
    insert_model_inference(&pool, tagged_model_id, tagged_inference_id, old_timestamp()).await;
    insert_model_inference_data(&pool, tagged_model_id, old_timestamp()).await;

    // Same chain for an untagged inference: its model data must survive.
    let untagged_inference_id = Uuid::now_v7();
    let untagged_model_id = Uuid::now_v7();
    insert_chat_inference(
        &pool,
        untagged_inference_id,
        serde_json::json!({}),
        old_timestamp(),
        None,
    )
    .await;
    insert_model_inference(
        &pool,
        untagged_model_id,
        untagged_inference_id,
        old_timestamp(),
    )
    .await;
    insert_model_inference_data(&pool, untagged_model_id, old_timestamp()).await;

    let run_id = run_cleanup_pass(
        &pool,
        cleanup_db::CLEANUP_TRIGGER_MANUAL,
        &CancellationToken::new(),
    )
    .await
    .expect("cleanup pass should succeed");

    expect_that!(
        count_model_data(&pool, tagged_model_id).await,
        eq(0),
        "model payload of a tagged inference should be deleted"
    );
    expect_that!(
        count_model_data(&pool, untagged_model_id).await,
        eq(1),
        "model payload of an untagged inference should survive"
    );

    tidy_up(
        &pool,
        rule.id,
        run_id,
        &tag_key,
        &[tagged_model_id, untagged_model_id],
    )
    .await;
}

/// A manual pass records a completed `cleanup_runs` row and one
/// `cleanup_run_steps` row per target table, with `total_rows` matching
/// `rows_deleted` and the actual deletions.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_cleanup_run_and_step_records() {
    let _guard = lock_cleanup().await;
    let conn = get_test_postgres().await;
    let pool = conn.get_pool().expect("Pool should be available").clone();

    let tag_key = unique_tag_key();
    let rule = cleanup_db::create_cleanup_rule(&pool, &tag_key, Some("yes"), 7, true)
        .await
        .expect("creating a cleanup rule should succeed");

    let tagged_id = Uuid::now_v7();
    insert_chat_inference(
        &pool,
        tagged_id,
        serde_json::json!({&tag_key: "yes"}),
        old_timestamp(),
        None,
    )
    .await;
    insert_chat_inference_data(&pool, tagged_id, old_timestamp()).await;

    let run_id = run_cleanup_pass(
        &pool,
        cleanup_db::CLEANUP_TRIGGER_MANUAL,
        &CancellationToken::new(),
    )
    .await
    .expect("cleanup pass should succeed");

    let (trigger, status, finished_at, error): (
        String,
        String,
        Option<DateTime<Utc>>,
        Option<String>,
    ) = sqlx::query_as(
        "SELECT trigger, status, finished_at, error FROM tensorzero.cleanup_runs WHERE id = $1",
    )
    .bind(run_id)
    .fetch_one(&pool)
    .await
    .expect("fetching the cleanup run should succeed");
    expect_that!(trigger.as_str(), eq("manual"), "trigger should be `manual`");
    expect_that!(status.as_str(), eq("completed"), "run should be completed");
    expect_that!(
        finished_at,
        some(anything()),
        "run should have `finished_at`"
    );
    expect_that!(error, none(), "run should have no error");

    let steps = steps_for_rule(&pool, run_id, rule.id).await;
    let mut table_names: Vec<&str> = steps.iter().map(|step| step.table_name.as_str()).collect();
    table_names.sort_unstable();
    let mut expected_tables: Vec<&str> = cleanup_db::CleanupTargetTable::ALL
        .iter()
        .map(|table| table.name())
        .collect();
    expected_tables.sort_unstable();
    assert_that!(
        table_names,
        container_eq(expected_tables),
        "one step should be recorded per target table"
    );

    for step in &steps {
        expect_that!(
            step.status.as_str(),
            eq("done"),
            "step for `{}` should be done",
            step.table_name
        );
        let expected = if step.table_name == "chat_inference_data" {
            1
        } else {
            0
        };
        expect_that!(
            step.total_rows,
            some(eq(expected)),
            "`total_rows` for `{}` should match the search result",
            step.table_name
        );
        expect_that!(
            step.rows_deleted,
            eq(expected),
            "`rows_deleted` for `{}` should match the actual deletions",
            step.table_name
        );
    }

    tidy_up(&pool, rule.id, run_id, &tag_key, &[]).await;
}

/// Deleting more than one chunk (1500 > 1000) exercises the chunked delete
/// loop; all rows are deleted and the step reports the full count.
/// (The name must not contain "batch": the e2e nextest profile filters those
/// names out.)
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_cleanup_many_rows_delete_in_chunks() {
    let _guard = lock_cleanup().await;
    let conn = get_test_postgres().await;
    let pool = conn.get_pool().expect("Pool should be available").clone();

    const ROW_COUNT: i64 = 1500;

    let tag_key = unique_tag_key();
    let rule = cleanup_db::create_cleanup_rule(&pool, &tag_key, Some("bulk"), 7, true)
        .await
        .expect("creating a cleanup rule should succeed");

    // One metadata row per payload row; the payload rows are backfilled from
    // the metadata rows so both share ids.
    sqlx::query(
        "INSERT INTO tensorzero.chat_inferences \
         (id, function_name, variant_name, episode_id, tags, created_at) \
         SELECT gen_random_uuid(), 'test_cleanup_fn', 'test_variant', gen_random_uuid(), \
                jsonb_build_object($1::TEXT, 'bulk'), now() - interval '10 days' \
         FROM generate_series(1, $2)",
    )
    .bind(&tag_key)
    .bind(ROW_COUNT)
    .execute(&pool)
    .await
    .expect("bulk-inserting metadata rows should succeed");
    sqlx::query(
        "INSERT INTO tensorzero.chat_inference_data \
         (id, input, output, inference_params, created_at) \
         SELECT id, '{}', '[]', '{}', now() - interval '10 days' \
         FROM tensorzero.chat_inferences WHERE tags ? $1",
    )
    .bind(&tag_key)
    .execute(&pool)
    .await
    .expect("bulk-inserting payload rows should succeed");

    let run_id = run_cleanup_pass(
        &pool,
        cleanup_db::CLEANUP_TRIGGER_MANUAL,
        &CancellationToken::new(),
    )
    .await
    .expect("cleanup pass should succeed");

    let remaining: i64 = sqlx::query_scalar(
        "SELECT COUNT(*)::BIGINT FROM tensorzero.chat_inference_data d \
         JOIN tensorzero.chat_inferences i ON i.id = d.id WHERE i.tags ? $1",
    )
    .bind(&tag_key)
    .fetch_one(&pool)
    .await
    .expect("counting remaining payload rows should succeed");
    expect_that!(
        remaining,
        eq(0),
        "all matching payload rows should be deleted"
    );

    let metadata_remaining: i64 = sqlx::query_scalar(
        "SELECT COUNT(*)::BIGINT FROM tensorzero.chat_inferences WHERE tags ? $1",
    )
    .bind(&tag_key)
    .fetch_one(&pool)
    .await
    .expect("counting remaining metadata rows should succeed");
    expect_that!(
        metadata_remaining,
        eq(ROW_COUNT),
        "metadata rows are never deleted by tag-based cleanup"
    );

    let steps = steps_for_rule(&pool, run_id, rule.id).await;
    let step = steps
        .iter()
        .find(|step| step.table_name == "chat_inference_data")
        .expect("a step for `chat_inference_data` should exist");
    expect_that!(
        step.total_rows,
        some(eq(ROW_COUNT)),
        "`total_rows` should cover the whole matching set"
    );
    expect_that!(
        step.rows_deleted,
        eq(ROW_COUNT),
        "`rows_deleted` should accumulate across chunks"
    );
    expect_that!(step.status.as_str(), eq("done"), "step should be done");

    tidy_up(&pool, rule.id, run_id, &tag_key, &[]).await;
}

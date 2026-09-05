// Modified by Delta-AI under Apache 2.0
mod experimentation_queries;
mod inference_protection;
mod inference_storage;
mod postgres_setup_tests;
// TODO(#5862): rename this to `batch_inference_writes` once nextest stops filtering on the name "batch".
mod pooled_inference_writes;
mod postgres_function_tests;
mod stored_configs;

use sqlx::Connection as _;

/// Arbitrary `pg_advisory_lock` key for tests that mutate the global
/// `tensorzero.retention_config` table. nextest runs every test in its own
/// process, so an in-process mutex cannot serialize them; a session-level
/// Postgres advisory lock serializes across processes against the same
/// e2e database.
const RETENTION_CONFIG_LOCK_KEY: i64 = 861_032;

/// Holds a session-level Postgres advisory lock serializing tests that mutate
/// the global `tensorzero.retention_config` table. The lock is released when
/// the guard is dropped (dropping the connection closes the session).
pub struct RetentionConfigLock {
    // Kept open for the lifetime of the guard; closing it releases the lock.
    _conn: sqlx::postgres::PgConnection,
}

/// Acquires the retention-config advisory lock, blocking until any other test
/// holding it has finished.
pub async fn lock_retention_config() -> RetentionConfigLock {
    let postgres_url = std::env::var("TENSORZERO_POSTGRES_URL")
        .expect("Environment variable TENSORZERO_POSTGRES_URL must be set");
    let mut conn = sqlx::postgres::PgConnection::connect(&postgres_url)
        .await
        .expect("connecting for the retention config lock should succeed");
    sqlx::query("SELECT pg_advisory_lock($1)")
        .bind(RETENTION_CONFIG_LOCK_KEY)
        .execute(&mut conn)
        .await
        .expect("acquiring the retention config advisory lock should succeed");
    RetentionConfigLock { _conn: conn }
}

/// Snapshots all rows of `tensorzero.retention_config` so a test can restore
/// the exact prior state after mutating it.
pub async fn snapshot_retention_config(pool: &sqlx::PgPool) -> Vec<(String, String)> {
    sqlx::query_as("SELECT key, value FROM tensorzero.retention_config")
        .fetch_all(pool)
        .await
        .expect("snapshotting `retention_config` should succeed")
}

/// Restores `tensorzero.retention_config` to a previous snapshot
/// (deletes all current rows, then re-inserts the snapshot).
pub async fn restore_retention_config(pool: &sqlx::PgPool, snapshot: &[(String, String)]) {
    sqlx::query("DELETE FROM tensorzero.retention_config")
        .execute(pool)
        .await
        .expect("clearing `retention_config` should succeed");
    for (key, value) in snapshot {
        sqlx::query("INSERT INTO tensorzero.retention_config (key, value) VALUES ($1, $2)")
            .bind(key)
            .bind(value)
            .execute(pool)
            .await
            .expect("restoring `retention_config` should succeed");
    }
}

// Modified by Delta-AI under Apache 2.0
//! Embedded worker for scheduled tag-based cleanup of Postgres payload tables.
//!
//! The worker is spawned by the gateway when `[gateway.cleanup]` is enabled.
//! It wakes on a configurable interval, on a manual trigger from the internal
//! API (`POST /internal/cleanup/run` via `cleanup_notify`), or on shutdown.
//! `enabled` / `interval_secs` are re-read from the live config on every
//! wakeup, so config hot-reloads take effect from the next pass. All rules are
//! read from Postgres at the start of each pass.

use std::sync::Arc;
use std::time::Duration;

use sqlx::PgPool;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::config::gateway::MIN_CLEANUP_INTERVAL_SECS;
use crate::db::postgres::cleanup as cleanup_db;
use crate::error::{Error, ErrorDetails};
use crate::utils::gateway::SwappableConfig;

/// Runs one cleanup pass: records a `cleanup_runs` row, applies every enabled
/// rule to all target tables, then marks the run completed or failed.
/// Returns the run id.
pub async fn run_cleanup_pass(
    pool: &PgPool,
    trigger: &str,
    shutdown_token: &CancellationToken,
) -> Result<Uuid, Error> {
    let run = cleanup_db::start_cleanup_run(pool, trigger).await?;
    let result = run_all_rules(pool, run.id, shutdown_token).await;
    match result {
        Ok(()) => {
            cleanup_db::finish_cleanup_run(
                pool,
                run.id,
                cleanup_db::CLEANUP_RUN_STATUS_COMPLETED,
                None,
            )
            .await?;
            tracing::info!("Cleanup run {} ({trigger}) completed", run.id);
        }
        Err(e) => {
            let message = e.to_string();
            if let Err(finish_error) = cleanup_db::finish_cleanup_run(
                pool,
                run.id,
                cleanup_db::CLEANUP_RUN_STATUS_FAILED,
                Some(&message),
            )
            .await
            {
                tracing::error!(
                    "Failed to mark cleanup run {} as failed: {finish_error}",
                    run.id
                );
            }
            return Err(e);
        }
    }
    Ok(run.id)
}

async fn run_all_rules(
    pool: &PgPool,
    run_id: Uuid,
    shutdown_token: &CancellationToken,
) -> Result<(), Error> {
    let rules = cleanup_db::list_enabled_cleanup_rules(pool).await?;
    for rule in rules {
        if shutdown_token.is_cancelled() {
            return Err(Error::new(ErrorDetails::InternalError {
                message: "Cleanup aborted: gateway is shutting down".to_string(),
            }));
        }
        cleanup_db::cleanup_rule(pool, &rule, run_id, shutdown_token).await?;
    }
    Ok(())
}

/// The cleanup worker loop. Never panics and never returns an error: a failed
/// pass is logged and the loop waits for the next wakeup. Runs until
/// `shutdown_token` is cancelled.
///
/// Takes a `SwappableConfig` handle rather than the full app state: `enabled`
/// and `interval_secs` are re-read from the latest config snapshot on every
/// wakeup, so config hot-reloads take effect from the next pass.
pub async fn cleanup_worker_loop(
    config: SwappableConfig,
    cleanup_notify: Arc<Notify>,
    pool: PgPool,
    shutdown_token: CancellationToken,
) {
    loop {
        // Read the interval before sleeping; a config hot-reload changing
        // `interval_secs` takes effect from the next pass.
        let interval_secs = config
            .load()
            .gateway
            .cleanup
            .interval_secs
            .max(MIN_CLEANUP_INTERVAL_SECS);
        let sleep = tokio::time::sleep(Duration::from_secs(interval_secs));
        tokio::pin!(sleep);
        let trigger = tokio::select! {
            () = shutdown_token.cancelled() => {
                tracing::info!("Cleanup worker shutting down");
                break;
            }
            () = &mut sleep => cleanup_db::CLEANUP_TRIGGER_SCHEDULE,
            () = cleanup_notify.notified() => cleanup_db::CLEANUP_TRIGGER_MANUAL,
        };

        // Re-read the config: it may have been hot-swapped while we slept.
        if !config.load().gateway.cleanup.enabled {
            tracing::debug!("Skipping cleanup pass: `gateway.cleanup.enabled` is false");
            continue;
        }

        if let Err(e) = run_cleanup_pass(&pool, trigger, &shutdown_token).await {
            tracing::error!("Cleanup pass ({trigger}) failed: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use googletest::prelude::*;

    #[gtest]
    fn trigger_and_status_constants_match_migration_checks() {
        // These values are constrained by CHECK constraints in the
        // `20261005000000_cleanup_tables.sql` migration; keep them in sync.
        expect_that!(cleanup_db::CLEANUP_TRIGGER_SCHEDULE, eq("schedule"));
        expect_that!(cleanup_db::CLEANUP_TRIGGER_MANUAL, eq("manual"));
        expect_that!(cleanup_db::CLEANUP_RUN_STATUS_RUNNING, eq("running"));
        expect_that!(cleanup_db::CLEANUP_RUN_STATUS_COMPLETED, eq("completed"));
        expect_that!(cleanup_db::CLEANUP_RUN_STATUS_FAILED, eq("failed"));
        expect_that!(cleanup_db::CLEANUP_STEP_STATUS_RUNNING, eq("running"));
        expect_that!(cleanup_db::CLEANUP_STEP_STATUS_DONE, eq("done"));
        expect_that!(cleanup_db::CLEANUP_STEP_STATUS_FAILED, eq("failed"));
    }
}

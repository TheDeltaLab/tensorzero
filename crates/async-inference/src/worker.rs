// Modified by Delta-AI under Apache 2.0
//! Worker startup for the async inference durable queue.

use anyhow::Result;
use durable::{Durable, Worker, WorkerOptions};
use sqlx::PgPool;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

use crate::state::AsyncInferenceState;
use crate::task::AsyncInferenceTask;

/// Configuration for the async inference worker.
pub struct AsyncInferenceWorkerConfig {
    /// Database pool for the durable task queue (shared with the gateway).
    pub pool: PgPool,
    /// Queue name for durable tasks (`gateway.async_inference.queue_name`).
    pub queue_name: String,
    /// Application state handed to each task execution.
    pub state: AsyncInferenceState,
    /// Options for the durable worker (poll interval, concurrency, etc.).
    pub worker_options: WorkerOptions,
}

/// Spawn the async inference worker as a background task.
///
/// The durable worker is started before spawning so configuration errors
/// (missing queue, missing migrations, database issues) surface at gateway
/// startup instead of failing silently in the background. The worker shuts
/// down when `cancel_token` is cancelled.
///
/// # Errors
///
/// Returns an error if the durable worker cannot be built or started.
pub async fn spawn_async_inference_worker(
    deferred_tasks: &TaskTracker,
    cancel_token: CancellationToken,
    config: AsyncInferenceWorkerConfig,
) -> Result<()> {
    let durable = Durable::builder()
        .pool(config.pool)
        .queue_name(config.queue_name)
        .register_instance(AsyncInferenceTask)?
        .build_with_state(config.state)
        .await?;

    let worker = durable.start_worker(config.worker_options).await?;

    deferred_tasks.spawn(async move {
        run_until_cancelled(worker, cancel_token).await;
    });
    Ok(())
}

/// Run the worker until the gateway shuts down.
async fn run_until_cancelled(worker: Worker, cancel_token: CancellationToken) {
    tokio::select! {
        () = cancel_token.cancelled() => {
            tracing::info!("Async inference worker received shutdown signal");
            worker.shutdown().await;
        }
    }
}

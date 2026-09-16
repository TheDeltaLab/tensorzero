// Modified by Delta-AI under Apache 2.0
//! The durable task that executes one async inference.

use std::borrow::Cow;
use std::future::Future;
use std::time::{Duration, Instant};

use anyhow::anyhow;
use durable::async_trait;
use durable::{StepState, Task, TaskContext, TaskResult};
use redis::AsyncCommands;
use redis::streams::StreamMaxlen;
use serde::Serialize;
use serde_json::Value;
use tensorzero_core::db::valkey::ValkeyConnection;
use tensorzero_core::endpoints::openai_compatible::async_inference::{
    ASYNC_INFERENCE_TASK_NAME, AsyncInferenceError, STREAM_FIELD_DATA, STREAM_FIELD_EVENT,
    STREAM_FIELD_MARKER, STREAM_MARKER_DONE, STREAM_MARKER_ERROR, STREAM_MAX_LEN,
    async_inference_stream_key, run_async_inference,
};
use tensorzero_core::endpoints::openai_compatible::async_inference_types::AsyncInferenceTaskParams;
use tensorzero_core::endpoints::openai_compatible::types::streaming::SerializedSseEvent;
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::state::AsyncInferenceState;

/// How often the lease is extended while an inference is running. Must be
/// comfortably below the worker's `claim_timeout` (120s by default).
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(30);

/// The durable task behind the async inference API.
///
/// Each execution runs the stored request via
/// [`run_async_inference`], relaying SSE frames to the task's Redis stream.
/// The final response (in the wire shape of the API the task was submitted
/// to) is the task output.
pub struct AsyncInferenceTask;

/// Params of the (single) `inference` step. `task_id` is included so the step
/// knows which Redis stream to write to and which task id to stamp onto the
/// inference's tags, without capturing variables.
#[derive(Serialize)]
struct InferenceStepParams {
    task_id: Uuid,
    params: AsyncInferenceTaskParams,
}

#[async_trait]
impl Task<AsyncInferenceState> for AsyncInferenceTask {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed(ASYNC_INFERENCE_TASK_NAME)
    }

    type Params = AsyncInferenceTaskParams;
    type Output = Value;

    async fn run(
        &self,
        params: Self::Params,
        mut ctx: TaskContext<AsyncInferenceState>,
        _state: AsyncInferenceState,
    ) -> TaskResult<Self::Output> {
        let task_id = ctx.task_id;
        let step_params = InferenceStepParams { task_id, params };
        let output = ctx
            .step("inference", step_params, execute_inference_step)
            .await?;
        Ok(output)
    }
}

/// Run the inference, relay SSE frames into the task's Redis stream, and
/// write the terminal `done`/`error` marker.
async fn execute_inference_step(
    step_params: InferenceStepParams,
    step_state: StepState<AsyncInferenceState>,
) -> anyhow::Result<Value> {
    let heartbeater = step_state.heartbeater;
    with_heartbeat(
        execute_with_relay(step_params, step_state.state),
        || async {
            heartbeater
                .heartbeat(None)
                .await
                .map_err(anyhow::Error::from)
        },
        HEARTBEAT_INTERVAL,
    )
    .await
}

/// Poll the heartbeat independently of Redis and provider work, including
/// initial cleanup and final stream flushing. Dropping `work` on lease loss
/// cancels the inference/relay together; no detached inference survives it.
async fn with_heartbeat<T, F, H, HF>(
    work: F,
    mut heartbeat: H,
    interval: Duration,
) -> anyhow::Result<T>
where
    F: Future<Output = anyhow::Result<T>>,
    H: FnMut() -> HF,
    HF: Future<Output = anyhow::Result<()>>,
{
    tokio::pin!(work);
    let mut ticks = tokio::time::interval(interval);
    ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    ticks.tick().await;
    loop {
        tokio::select! {
            biased;
            _ = ticks.tick() => heartbeat().await?,
            result = &mut work => return result,
        }
    }
}

async fn execute_with_relay(
    step_params: InferenceStepParams,
    state: AsyncInferenceState,
) -> anyhow::Result<Value> {
    let key = async_inference_stream_key(step_params.task_id);
    let mut writer = StreamWriter::new(state.valkey.clone(), key.clone(), state.stream_ttl);
    // Do not append to a previous attempt if cleanup failed. The final task
    // result remains authoritative if the stream fails after inference starts.
    // A cleanup failure occurs before any provider call, so retrying is safe.
    let mut conn = state.valkey.clone();
    conn.del::<_, ()>(&key).await.map_err(|e| {
        anyhow!(
            "Cannot clear the previous async inference stream before starting a new attempt: {e}"
        )
    })?;

    // Backpressure bounds retained events while a Redis operation is slow.
    // The relay and model are polled independently, under the heartbeat above.
    let (event_tx, mut event_rx) = mpsc::channel::<SerializedSseEvent>(64);
    let app_state = state.app_state.load_latest();
    let inference = run_async_inference(
        &app_state,
        step_params.task_id,
        step_params.params,
        event_tx,
    );
    let relay = async {
        let mut complete = true;
        while let Some(frame) = event_rx.recv().await {
            if complete && let Err(e) = writer.add_frame(&frame).await {
                // An XADD timeout has an ambiguous outcome: never replay it.
                // Drain subsequent events without publishing an incomplete
                // sequence as success. Polling still returns the full result.
                tracing::warn!("Async inference event relay interrupted: {e}");
                complete = false;
            }
        }
        complete
    };
    let (result, complete_stream) = tokio::join!(inference, relay);
    if !complete_stream {
        writer.write_terminal_marker(STREAM_MARKER_ERROR, Some(
            serde_json::json!({"error": {"message": "Async event stream interrupted; fetch the complete result from the task status endpoint"}}).to_string(),
        )).await;
    }
    match result {
        Ok(response) => {
            if complete_stream {
                writer.write_terminal_marker(STREAM_MARKER_DONE, None).await;
            }
            Ok(response)
        }
        Err(error) => {
            let AsyncInferenceError { message, body } = error;
            if complete_stream {
                writer
                    .write_terminal_marker(STREAM_MARKER_ERROR, Some(body.to_string()))
                    .await;
            }
            Err(anyhow!(message))
        }
    }
}

/// Writes SSE frames and terminal markers to a task's Redis stream, keeping
/// the stream's TTL fresh while the task is running.
struct StreamWriter {
    conn: ValkeyConnection,
    key: String,
    ttl: Duration,
    last_expire: Option<Instant>,
}

impl StreamWriter {
    fn new(conn: ValkeyConnection, key: String, ttl: Duration) -> Self {
        Self {
            conn,
            key,
            ttl,
            last_expire: None,
        }
    }

    /// XADD one SSE frame, refreshing the stream TTL at most every `ttl / 2`.
    async fn add_frame(&mut self, frame: &SerializedSseEvent) -> redis::RedisResult<()> {
        let entries: Vec<(&str, &str)> = match &frame.event {
            Some(event) => vec![
                (STREAM_FIELD_EVENT, event.as_str()),
                (STREAM_FIELD_DATA, frame.data.as_str()),
            ],
            None => vec![(STREAM_FIELD_DATA, frame.data.as_str())],
        };
        let _: Option<String> = self
            .conn
            .xadd_maxlen(
                &self.key,
                StreamMaxlen::Approx(STREAM_MAX_LEN),
                "*",
                &entries,
            )
            .await?;
        self.maybe_refresh_ttl().await;
        Ok(())
    }

    /// XADD the terminal marker and set the final TTL. Errors are logged, not
    /// propagated: the task result itself is the source of truth, the stream
    /// is best-effort.
    async fn write_terminal_marker(&mut self, marker: &str, data: Option<String>) {
        let data = data.unwrap_or_default();
        let entries = [
            (STREAM_FIELD_MARKER, marker),
            (STREAM_FIELD_DATA, data.as_str()),
        ];
        let result: redis::RedisResult<Option<String>> = self
            .conn
            .xadd_maxlen(
                &self.key,
                StreamMaxlen::Approx(STREAM_MAX_LEN),
                "*",
                &entries,
            )
            .await;
        if let Err(e) = result {
            tracing::warn!(
                "Failed to write `{marker}` marker to async inference stream `{}`: {e}",
                self.key
            );
        }
        self.expire().await;
    }

    async fn maybe_refresh_ttl(&mut self) {
        let refresh_after = self.ttl / 2;
        if self
            .last_expire
            .is_some_and(|at| at.elapsed() < refresh_after)
        {
            return;
        }
        self.expire().await;
    }

    async fn expire(&mut self) {
        let result: redis::RedisResult<bool> =
            self.conn.expire(&self.key, self.ttl.as_secs() as i64).await;
        match result {
            Ok(_) => self.last_expire = Some(Instant::now()),
            Err(e) => {
                tracing::warn!(
                    "Failed to refresh TTL on async inference stream `{}`: {e}",
                    self.key
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use googletest::prelude::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    #[gtest]
    #[tokio::test(start_paused = true)]
    async fn blocked_work_does_not_block_heartbeats() {
        let count = AtomicUsize::new(0);
        let result = with_heartbeat(
            async {
                tokio::time::sleep(Duration::from_secs(245)).await;
                Ok(7)
            },
            || async {
                count.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
            HEARTBEAT_INTERVAL,
        )
        .await
        .expect("work completes despite exceeding the original lease");
        expect_that!(result, eq(7));
        expect_that!(count.load(Ordering::SeqCst), eq(8));
    }

    struct DropGuard(Arc<AtomicBool>);
    impl Drop for DropGuard {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[gtest]
    #[tokio::test(start_paused = true)]
    async fn lost_lease_cancels_inference_and_relay() {
        let dropped = Arc::new(AtomicBool::new(false));
        let guard = DropGuard(dropped.clone());
        let result: anyhow::Result<()> = with_heartbeat(
            async move {
                let _guard = guard;
                std::future::pending().await
            },
            || async { Err(anyhow!("lease lost")) },
            HEARTBEAT_INTERVAL,
        )
        .await;
        expect_that!(result.is_err(), eq(true));
        expect_that!(
            dropped.load(Ordering::SeqCst),
            eq(true),
            "no detached inference after cancellation"
        );
    }
}

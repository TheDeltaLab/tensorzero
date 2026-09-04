// Modified by Delta-AI under Apache 2.0
//! The durable task that executes one async inference.

use std::borrow::Cow;
use std::time::{Duration, Instant};

use anyhow::anyhow;
use durable::async_trait;
use durable::{StepState, Task, TaskContext, TaskResult};
use redis::AsyncCommands;
use redis::aio::ConnectionManager;
use redis::streams::StreamMaxlen;
use serde::Serialize;
use serde_json::Value;
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
        state: AsyncInferenceState,
    ) -> TaskResult<Self::Output> {
        let task_id = ctx.task_id;
        let key = async_inference_stream_key(task_id);

        // Clear stale events from a previous attempt before re-running.
        let mut conn = state.valkey.clone();
        if let Err(e) = conn.del::<_, ()>(&key).await {
            tracing::warn!("Failed to clear async inference event stream `{key}`: {e}");
        }

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
    let state = step_state.state;
    let key = async_inference_stream_key(step_params.task_id);
    let mut writer = StreamWriter::new(state.valkey.clone(), key, state.stream_ttl);

    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<SerializedSseEvent>();
    let app_state = state.app_state.load_latest();
    let mut inference = Box::pin(run_async_inference(
        &app_state,
        step_params.task_id,
        step_params.params,
        event_tx,
    ));

    // Drive the inference while relaying frames and refreshing the lease.
    // `biased` so a completed inference is observed before a closed channel.
    let heartbeater = step_state.heartbeater;
    let mut heartbeat_interval = tokio::time::interval(HEARTBEAT_INTERVAL);
    heartbeat_interval.tick().await;
    let result = loop {
        tokio::select! {
            biased;
            result = &mut inference => break result,
            frame = event_rx.recv() => {
                // The sender is dropped when the inference future finishes;
                // the next `biased` poll of `inference` observes its result.
                if let Some(frame) = frame
                    && let Err(e) = writer.add_frame(&frame).await
                {
                    tracing::warn!("Failed to relay async inference event: {e}");
                }
            }
            _ = heartbeat_interval.tick() => {
                if let Err(e) = heartbeater.heartbeat(None).await {
                    // Cancelled (or the lease was lost): abort the inference by
                    // dropping its future and fail this run.
                    drop(inference);
                    return Err(anyhow!("Async inference task heartbeat failed: {e}"));
                }
            }
        }
    };

    // Flush any frames queued between the last relay poll and completion.
    while let Ok(frame) = event_rx.try_recv() {
        if let Err(e) = writer.add_frame(&frame).await {
            tracing::warn!("Failed to relay async inference event: {e}");
        }
    }

    match result {
        Ok(response) => {
            writer.write_terminal_marker(STREAM_MARKER_DONE, None).await;
            Ok(response)
        }
        Err(error) => {
            let AsyncInferenceError { message, body } = error;
            writer
                .write_terminal_marker(STREAM_MARKER_ERROR, Some(body.to_string()))
                .await;
            Err(anyhow!(message))
        }
    }
}

/// Writes SSE frames and terminal markers to a task's Redis stream, keeping
/// the stream's TTL fresh while the task is running.
struct StreamWriter {
    conn: ConnectionManager,
    key: String,
    ttl: Duration,
    last_expire: Option<Instant>,
}

impl StreamWriter {
    fn new(conn: ConnectionManager, key: String, ttl: Duration) -> Self {
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

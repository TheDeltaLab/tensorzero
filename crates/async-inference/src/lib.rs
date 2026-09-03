// Modified by Delta-AI under Apache 2.0
#![recursion_limit = "256"]
//! Durable worker for the async inference API.
//!
//! Executes `async_inference` tasks enqueued by the gateway's
//! `POST .../async` submit endpoints: runs the inference against the latest
//! gateway config, relays SSE frames into a per-task Redis stream (for
//! `GET /v1/async_tasks/{task_id}/stream`), and returns the final response as
//! the task output (for `GET /v1/async_tasks/{task_id}`).

mod state;
mod task;
mod worker;

pub use state::AsyncInferenceState;
pub use task::AsyncInferenceTask;
pub use worker::{AsyncInferenceWorkerConfig, spawn_async_inference_worker};

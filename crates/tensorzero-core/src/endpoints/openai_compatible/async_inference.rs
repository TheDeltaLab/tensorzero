// Modified by Delta-AI under Apache 2.0
//! Async inference API.
//!
//! Submit endpoints (`POST .../async` variants of the OpenAI chat completions,
//! OpenAI responses, and Anthropic messages APIs) validate the request
//! synchronously, enqueue a durable task on the `async_inference` queue, and
//! return `202 Accepted` with a `task_id`. The task is executed by the
//! `async-inference` worker crate.
//!
//! - `GET /v1/async_tasks/{task_id}` polls the task status and, once completed,
//!   returns the final response in the shape of the API the task was submitted
//!   to.
//! - `GET /v1/async_tasks/{task_id}/stream` replays and follows the SSE event
//!   stream via a Redis stream written by the worker
//!   (`{ASYNC_INFERENCE_STREAM_KEY_PREFIX}{task_id}`), so clients can attach to
//!   a running (or recently finished) task.

use std::collections::{BTreeMap, HashMap};
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use axum::Extension;
use axum::Json;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use chrono::Utc;
use durable_tools_spawn::{SpawnClient, SpawnError, SpawnOptions, TaskPollResult, TaskStatus};
use futures::Stream;
use futures::stream::StreamExt;
use redis::AsyncCommands;
use redis::aio::ConnectionManager;
use redis::streams::{StreamId, StreamRangeReply, StreamReadOptions, StreamReadReply};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use tensorzero_auth::middleware::RequestApiKeyExtension;
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::endpoints::inference::{
    ChatInferenceResponse, ChatInferenceResponseChunk, InferenceOutput, InferenceResponse,
    InferenceResponseChunk, InferenceStream, JsonInferenceResponse, JsonInferenceResponseChunk,
};
use crate::error::{Error, ErrorDetails};
use crate::observability_tags::{API_KEY_PUBLIC_ID_TAG, ASYNC_TAG, ASYNC_TASK_ID_TAG};
use crate::utils::gateway::{AppState, AppStateData};

use super::anthropic_messages::{
    AnthropicMessagesParams, anthropic_from_inference, execute_anthropic, prepare_anthropic_sse,
    validate_anthropic_request,
};
use super::async_inference_types::{
    AsyncInferenceApiKind, AsyncInferenceLaunchResponse, AsyncInferenceTaskParams,
    AsyncTaskStatusResponse,
};
use super::infer::{execute_openai_compatible, validate_openai_compatible_request};
use super::responses::prepare_serialized_openai_responses_events;
use super::types::chat_completions::{OpenAICompatibleParams, OpenAICompatibleResponse};
use super::types::responses::{OpenAICompatibleResponsesParams, OpenAICompatibleResponsesResponse};
use super::types::streaming::{SerializedSseEvent, prepare_serialized_openai_compatible_events};
use super::{OpenAICompatibleError, OpenAIStructuredJson};

/// Name of the durable task registered by the async inference worker.
pub const ASYNC_INFERENCE_TASK_NAME: &str = "async_inference";

/// Redis stream key prefix for async inference SSE event streams; the full key
/// is `{ASYNC_INFERENCE_STREAM_KEY_PREFIX}{task_id}`.
pub const ASYNC_INFERENCE_STREAM_KEY_PREFIX: &str = "tensorzero:async_inference:";

/// Redis stream field carrying the terminal marker (`done` / `error`).
pub const STREAM_FIELD_MARKER: &str = "marker";
/// Redis stream field carrying the SSE `event:` name.
pub const STREAM_FIELD_EVENT: &str = "event";
/// Redis stream field carrying the SSE `data:` payload.
pub const STREAM_FIELD_DATA: &str = "data";
/// Marker value written when the task completed successfully.
pub const STREAM_MARKER_DONE: &str = "done";
/// Marker value written when the task failed; the `data` field carries the
/// error body.
pub const STREAM_MARKER_ERROR: &str = "error";

/// How long an XREAD call blocks waiting for new stream entries before the
/// handler re-checks the task status in Postgres. In practice the valkey
/// client's 500ms default response timeout fires first on an idle stream;
/// the follow loop treats that client-side timeout as an empty read.
const XREAD_BLOCK_MS: usize = 5000;

/// Upper bound on the number of entries kept in a task's Redis stream.
pub const STREAM_MAX_LEN: usize = 10000;

/// Redis stream key for a task's SSE event stream.
pub fn async_inference_stream_key(task_id: Uuid) -> String {
    format!("{ASYNC_INFERENCE_STREAM_KEY_PREFIX}{task_id}")
}

// ---------------------------------------------------------------------------
// Submit handlers
// ---------------------------------------------------------------------------

/// `POST /v1/chat/completions/async` (and `/openai/v1/...`).
pub async fn chat_completions_async_handler(
    State(state): AppState,
    api_key_ext: Option<Extension<RequestApiKeyExtension>>,
    headers: HeaderMap,
    OpenAIStructuredJson(body): OpenAIStructuredJson<Value>,
) -> Result<Response, OpenAICompatibleError> {
    submit_async_inference(
        &state,
        api_key_ext,
        &headers,
        AsyncInferenceApiKind::Chat,
        body,
    )
    .await
}

/// `POST /v1/responses/async` (and `/openai/v1/...`).
pub async fn responses_async_handler(
    State(state): AppState,
    api_key_ext: Option<Extension<RequestApiKeyExtension>>,
    headers: HeaderMap,
    OpenAIStructuredJson(body): OpenAIStructuredJson<Value>,
) -> Result<Response, OpenAICompatibleError> {
    submit_async_inference(
        &state,
        api_key_ext,
        &headers,
        AsyncInferenceApiKind::Responses,
        body,
    )
    .await
}

/// `POST /v1/messages/async` (and `/openai/v1/messages/async`,
/// `/anthropic/v1/messages/async`).
pub async fn messages_async_handler(
    State(state): AppState,
    api_key_ext: Option<Extension<RequestApiKeyExtension>>,
    headers: HeaderMap,
    OpenAIStructuredJson(body): OpenAIStructuredJson<Value>,
) -> Result<Response, OpenAICompatibleError> {
    submit_async_inference(
        &state,
        api_key_ext,
        &headers,
        AsyncInferenceApiKind::Messages,
        body,
    )
    .await
}

/// Validate the request body, capture compatibility headers and the API key's
/// public ID, and enqueue the durable task.
async fn submit_async_inference(
    state: &AppStateData,
    api_key_ext: Option<Extension<RequestApiKeyExtension>>,
    headers: &HeaderMap,
    api_kind: AsyncInferenceApiKind,
    body: Value,
) -> Result<Response, OpenAICompatibleError> {
    let spawn_client = require_spawn_client(state)?;

    // Validate synchronously so obvious request errors surface at submit time
    // rather than as a failed task.
    validate_submit_body(api_kind, headers, &body)?;

    let task_params = AsyncInferenceTaskParams {
        api_kind,
        request: body,
        headers: capture_headers(headers),
        api_key_public_id: api_key_public_id(api_key_ext.as_ref()),
    };
    let params_value = serde_json::to_value(&task_params).map_err(|e| {
        Error::new(ErrorDetails::InternalError {
            message: format!("Failed to serialize async inference task params: {e}"),
        })
    })?;

    let spawned = spawn_client
        .spawn_task_by_name(
            ASYNC_INFERENCE_TASK_NAME,
            params_value,
            SpawnOptions::default(),
        )
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::InternalError {
                message: format!("Failed to enqueue async inference task: {e}"),
            })
        })?;

    Ok((
        StatusCode::ACCEPTED,
        Json(AsyncInferenceLaunchResponse {
            task_id: spawned.task_id,
        }),
    )
        .into_response())
}

/// Validate a submit body the same way the synchronous endpoint would, with
/// `stream` forced on (execution is always streaming internally so events can
/// be relayed).
fn validate_submit_body(
    api_kind: AsyncInferenceApiKind,
    headers: &HeaderMap,
    body: &Value,
) -> Result<(), Error> {
    match api_kind {
        AsyncInferenceApiKind::Chat => {
            let mut params: OpenAICompatibleParams = deserialize_body(body)?;
            params.stream = Some(true);
            validate_openai_compatible_request(headers, params)
                .map_err(|rejection| rejection.error)?;
        }
        AsyncInferenceApiKind::Responses => {
            let responses_params: OpenAICompatibleResponsesParams = deserialize_body(body)?;
            let mut params = responses_params.into_chat_params()?;
            params.stream = Some(true);
            validate_openai_compatible_request(headers, params)
                .map_err(|rejection| rejection.error)?;
        }
        AsyncInferenceApiKind::Messages => {
            let mut params: AnthropicMessagesParams = deserialize_body(body)?;
            params.stream = Some(true);
            validate_anthropic_request(headers, params)?;
        }
    }
    Ok(())
}

/// Deserialize a raw JSON body into a request params type, mirroring the sync
/// endpoints' body-parse error (`400` with the serde path).
fn deserialize_body<T: DeserializeOwned>(body: &Value) -> Result<T, Error> {
    serde_path_to_error::deserialize(body).map_err(|e| {
        Error::new(ErrorDetails::JsonRequest {
            message: e.to_string(),
        })
    })
}

/// Capture the headers needed to reconstruct the request context in the
/// worker: Synapse compatibility headers, TensorZero headers, and the request
/// ID. Credentials (`authorization`, API keys) are never captured.
fn capture_headers(headers: &HeaderMap) -> BTreeMap<String, String> {
    headers
        .iter()
        .filter_map(|(name, value)| {
            let name = name.as_str().to_ascii_lowercase();
            let keep = name.starts_with("x-synapse-")
                || name.starts_with("x-tensorzero-")
                || name == "x-request-id";
            if !keep {
                return None;
            }
            let value = value.to_str().ok()?.to_string();
            Some((name, value))
        })
        .collect()
}

/// Rebuild a [`HeaderMap`] from the headers captured at submit time.
fn rebuild_headers(captured: &BTreeMap<String, String>) -> Result<HeaderMap, AsyncInferenceError> {
    let mut headers = HeaderMap::new();
    for (name, value) in captured {
        let name = name.parse::<HeaderName>().map_err(|e| {
            AsyncInferenceError::internal(format!(
                "Stored async inference header name `{name}` is invalid: {e}"
            ))
        })?;
        let value = HeaderValue::from_str(value).map_err(|e| {
            AsyncInferenceError::internal(format!(
                "Stored async inference header value for `{name}` is invalid: {e}"
            ))
        })?;
        headers.append(name, value);
    }
    Ok(headers)
}

/// Public ID of the authenticated API key, if the submit request carried one.
/// Only the public ID is stored in the task params, never the secret key.
fn api_key_public_id(api_key_ext: Option<&Extension<RequestApiKeyExtension>>) -> Option<String> {
    api_key_ext.map(|ext| ext.0.api_key.get_public_id().to_string())
}

/// Stamp the submitting API key's public ID onto the inference tags, mirroring
/// the sync path (`inference()` inserts the same tag when `api_key_ext` is
/// present), so async-written inference rows appear in API-key-filtered views.
fn insert_api_key_public_id_tag(
    tags: &mut HashMap<String, String>,
    api_key_public_id: Option<&str>,
) {
    let Some(public_id) = api_key_public_id else {
        return;
    };
    tags.insert(API_KEY_PUBLIC_ID_TAG.to_string(), public_id.into());
}

/// Mark the inference as async-executed, stamped with the durable task's UUID.
/// Inserted after the request body's tags so the worker's values win over any
/// caller-supplied tags with the same keys.
fn insert_async_tags(tags: &mut HashMap<String, String>, task_id: Uuid) {
    tags.insert(ASYNC_TAG.to_string(), "true".to_string());
    tags.insert(ASYNC_TASK_ID_TAG.to_string(), task_id.to_string());
}

fn require_spawn_client(state: &AppStateData) -> Result<Arc<SpawnClient>, OpenAICompatibleError> {
    state
        .async_inference_spawn_client
        .clone()
        .ok_or_else(|| {
            OpenAICompatibleError(Error::new(ErrorDetails::Config {
                message: "Async inference is not enabled. Set `gateway.async_inference.enabled = true` and configure Postgres to use the async inference API.".to_string(),
            }))
        })
}

// ---------------------------------------------------------------------------
// Status handler
// ---------------------------------------------------------------------------

/// `GET /v1/async_tasks/{task_id}` (and `/openai/v1/...`).
pub async fn get_async_task_handler(
    State(state): AppState,
    Path(task_id): Path<Uuid>,
) -> Result<Json<AsyncTaskStatusResponse>, OpenAICompatibleError> {
    let spawn_client = require_spawn_client(&state)?;
    let poll = poll_task(&spawn_client, task_id).await?;
    Ok(Json(
        task_status_response(&spawn_client, task_id, poll).await,
    ))
}

/// Poll the durable task, mapping a missing task to a 404.
async fn poll_task(
    spawn_client: &SpawnClient,
    task_id: Uuid,
) -> Result<TaskPollResult, OpenAICompatibleError> {
    spawn_client
        .get_task_result(task_id)
        .await
        .map_err(|error| match error {
            SpawnError::TaskNotFound(_) => Error::new(ErrorDetails::RouteNotFound {
                path: format!("/v1/async_tasks/{task_id}"),
                method: "GET".to_string(),
            })
            .into(),
            other => Error::new(ErrorDetails::InternalError {
                message: format!("Failed to fetch async inference task `{task_id}`: {other}"),
            })
            .into(),
        })
}

async fn task_status_response(
    spawn_client: &SpawnClient,
    task_id: Uuid,
    poll: TaskPollResult,
) -> AsyncTaskStatusResponse {
    match poll.status {
        TaskStatus::Pending | TaskStatus::Sleeping => {
            let queue_position = match spawn_client.get_queue_position(task_id).await {
                Ok(position) => position,
                Err(e) => {
                    tracing::warn!(
                        "Failed to compute queue position for async inference task `{task_id}`: {e}"
                    );
                    None
                }
            };
            AsyncTaskStatusResponse::Queued {
                task_id,
                queue_position,
            }
        }
        TaskStatus::Running => {
            let (started_at, elapsed_ms) = match spawn_client.get_task_timing(task_id).await {
                Ok(Some(timing)) => match timing.first_started_at {
                    Some(started) => {
                        let elapsed = Utc::now()
                            .signed_duration_since(started)
                            .num_milliseconds()
                            .max(0) as u64;
                        (Some(started), Some(elapsed))
                    }
                    None => (None, None),
                },
                Ok(None) => (None, None),
                Err(e) => {
                    tracing::warn!(
                        "Failed to fetch timing for async inference task `{task_id}`: {e}"
                    );
                    (None, None)
                }
            };
            AsyncTaskStatusResponse::Running {
                task_id,
                started_at,
                elapsed_ms,
            }
        }
        TaskStatus::Completed => AsyncTaskStatusResponse::Completed {
            task_id,
            response: poll.result.unwrap_or(Value::Null),
        },
        TaskStatus::Failed => AsyncTaskStatusResponse::Failed {
            task_id,
            error: poll.error,
        },
        TaskStatus::Cancelled => AsyncTaskStatusResponse::Cancelled {
            task_id,
            error: poll.error,
        },
    }
}

// ---------------------------------------------------------------------------
// Stream handler
// ---------------------------------------------------------------------------

/// `GET /v1/async_tasks/{task_id}/stream` (and `/openai/v1/...`).
///
/// Replays the events the worker has written so far, then follows the Redis
/// stream live until the worker writes a terminal `done`/`error` marker (or
/// the task reaches a terminal state without one, e.g. after a worker crash).
pub async fn stream_async_task_handler(
    State(state): AppState,
    Path(task_id): Path<Uuid>,
) -> Result<Response, OpenAICompatibleError> {
    let spawn_client = require_spawn_client(&state)?;
    let conn = state
        .valkey_connection_info
        .get_connection()
        .cloned()
        .ok_or_else(|| {
            OpenAICompatibleError(Error::new(ErrorDetails::Config {
                message: "Async inference streaming requires Valkey to be configured.".to_string(),
            }))
        })?;

    let poll = poll_task(&spawn_client, task_id).await?;
    let key = async_inference_stream_key(task_id);

    let mut replay_conn = conn.clone();
    let replay: StreamRangeReply = replay_conn.xrange(&key, "-", "+").await.map_err(|e| {
        Error::new(ErrorDetails::InternalError {
            message: format!(
                "Failed to read async inference event stream for task `{task_id}`: {e}"
            ),
        })
    })?;

    if replay.ids.is_empty() && poll.status.is_terminal() {
        return Ok(async_stream_gone_response(task_id));
    }

    let event_stream = async_task_event_stream(spawn_client, conn, key, task_id, replay);
    Ok(
        Sse::new(event_stream.take_until(state.shutdown_token.clone().cancelled_owned()))
            .keep_alive(KeepAlive::new())
            .into_response(),
    )
}

/// 410 Gone response for tasks whose event stream is gone (expired or never
/// written because the worker crashed before the first event).
fn async_stream_gone_response(task_id: Uuid) -> Response {
    let body = json!({
        "error": {
            "message": format!(
                "Async inference task `{task_id}` has already finished and its event stream is no longer available. Fetch the final result from `GET /v1/async_tasks/{task_id}`."
            )
        }
    });
    (StatusCode::GONE, Json(body)).into_response()
}

/// What to do with one entry of the task's Redis stream.
#[derive(Debug)]
enum StreamEntryAction {
    /// Yield the event and continue.
    Event(Event),
    /// Yield the event, then end the stream (terminal `error` marker).
    TerminalEvent(Event),
    /// End the stream without yielding (terminal `done` marker).
    End,
}

fn parse_stream_entry(entry: &StreamId) -> Result<StreamEntryAction, Error> {
    let read_field = |field: &str| -> Result<Option<String>, Error> {
        entry
            .map
            .get(field)
            .map(|value| {
                redis::from_redis_value::<String>(value.clone()).map_err(|e| {
                    Error::new(ErrorDetails::InternalError {
                        message: format!(
                            "Malformed async inference stream entry `{}` (field `{field}`): {e}",
                            entry.id
                        ),
                    })
                })
            })
            .transpose()
    };
    let marker = read_field(STREAM_FIELD_MARKER)?;
    // The `done` marker carries no payload.
    if marker.as_deref() == Some(STREAM_MARKER_DONE) {
        return Ok(StreamEntryAction::End);
    }
    let data = read_field(STREAM_FIELD_DATA)?.ok_or_else(|| {
        Error::new(ErrorDetails::InternalError {
            message: format!(
                "Malformed async inference stream entry `{}`: missing `{STREAM_FIELD_DATA}` field",
                entry.id
            ),
        })
    })?;
    match marker.as_deref() {
        Some(STREAM_MARKER_ERROR) => Ok(StreamEntryAction::TerminalEvent(
            Event::default().event("error").data(data),
        )),
        Some(other) => Err(Error::new(ErrorDetails::InternalError {
            message: format!(
                "Malformed async inference stream entry `{}`: unknown marker `{other}`",
                entry.id
            ),
        })),
        None => {
            let event = read_field(STREAM_FIELD_EVENT)?;
            Ok(StreamEntryAction::Event(
                SerializedSseEvent::new(event, data).into_event(),
            ))
        }
    }
}

/// Build the SSE event stream: replay existing entries, then follow the Redis
/// stream with blocking XREADs until a terminal marker arrives or the task
/// reaches a terminal state in Postgres.
fn async_task_event_stream(
    spawn_client: Arc<SpawnClient>,
    mut conn: ConnectionManager,
    key: String,
    task_id: Uuid,
    replay: StreamRangeReply,
) -> impl Stream<Item = Result<Event, Error>> + use<> {
    async_stream::try_stream! {
        let mut last_id = "0-0".to_string();
        let mut finished = false;

        for entry in replay.ids {
            last_id = entry.id.clone();
            match parse_stream_entry(&entry)? {
                StreamEntryAction::Event(event) => yield event,
                StreamEntryAction::TerminalEvent(event) => {
                    yield event;
                    finished = true;
                    break;
                }
                StreamEntryAction::End => {
                    finished = true;
                    break;
                }
            }
        }

        while !finished {
            let options = StreamReadOptions::default().block(XREAD_BLOCK_MS);
            let reply: Option<StreamReadReply> = match conn
                .xread_options(&[key.as_str()], &[last_id.as_str()], &options)
                .await
            {
                Ok(reply) => reply,
                // The shared valkey `ConnectionManager` times commands out
                // after 500ms by default, below `XREAD_BLOCK_MS`, so an idle
                // blocking read always ends in a client-side timeout. Treat
                // that like an empty blocking read: fall through to the task
                // status re-check below.
                Err(e) if e.is_timeout() => None,
                Err(e) => Err(Error::new(ErrorDetails::InternalError {
                    message: format!(
                        "Failed to follow async inference event stream for task `{task_id}`: {e}"
                    ),
                }))?,
            };

            let Some(reply) = reply else {
                // Block timed out with no new entries. Re-check the task status:
                // if the worker crashed before writing a terminal marker, the
                // task is terminal in Postgres and we end the stream here.
                let poll = spawn_client.get_task_result(task_id).await.map_err(|e| {
                    Error::new(ErrorDetails::InternalError {
                        message: format!("Failed to poll async inference task `{task_id}`: {e}"),
                    })
                })?;
                if poll.status.is_terminal() {
                    break;
                }
                continue;
            };

            for stream_key in reply.keys {
                for entry in stream_key.ids {
                    last_id = entry.id.clone();
                    match parse_stream_entry(&entry)? {
                        StreamEntryAction::Event(event) => yield event,
                        StreamEntryAction::TerminalEvent(event) => {
                            yield event;
                            finished = true;
                        }
                        StreamEntryAction::End => finished = true,
                    }
                    if finished {
                        break;
                    }
                }
                if finished {
                    break;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Worker-side execution (used by the `async-inference` worker crate)
// ---------------------------------------------------------------------------

/// Error type returned by [`run_async_inference`]. The `body` is the
/// OpenAI-shaped error payload written to the Redis stream's terminal `error`
/// marker; the `message` becomes the durable task's failure reason.
#[derive(Debug)]
pub struct AsyncInferenceError {
    pub message: String,
    pub body: Value,
}

impl AsyncInferenceError {
    pub fn from_error(error: &Error, include_raw_response: bool) -> Self {
        Self {
            message: error.to_string(),
            body: error.build_streaming_error_event(true, include_raw_response),
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        let message = message.into();
        Self {
            body: json!({"error": {"message": message}}),
            message,
        }
    }
}

impl std::fmt::Display for AsyncInferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for AsyncInferenceError {}

/// Run one async inference task: re-validate the stored request, execute the
/// inference (always streaming internally, with `include_aggregated_response`
/// so the last chunk carries the full output), forward each SSE frame to
/// `event_tx`, and return the final response in the wire shape of the API the
/// task was submitted to.
///
/// Send errors on `event_tx` are ignored: the event stream is best-effort,
/// while the final response is the durable task output.
pub async fn run_async_inference(
    state: &AppStateData,
    task_id: Uuid,
    params: AsyncInferenceTaskParams,
    event_tx: mpsc::UnboundedSender<SerializedSseEvent>,
) -> Result<Value, AsyncInferenceError> {
    let headers = rebuild_headers(&params.headers)?;
    let api_key_public_id = params.api_key_public_id.as_deref();
    match params.api_kind {
        AsyncInferenceApiKind::Chat => {
            Box::pin(run_openai_style(
                state,
                &headers,
                params.request,
                OpenAIStyle::Chat,
                task_id,
                api_key_public_id,
                event_tx,
            ))
            .await
        }
        AsyncInferenceApiKind::Responses => {
            Box::pin(run_openai_style(
                state,
                &headers,
                params.request,
                OpenAIStyle::Responses,
                task_id,
                api_key_public_id,
                event_tx,
            ))
            .await
        }
        AsyncInferenceApiKind::Messages => {
            Box::pin(run_messages_style(
                state,
                &headers,
                params.request,
                task_id,
                api_key_public_id,
                event_tx,
            ))
            .await
        }
    }
}

#[derive(Clone, Copy)]
enum OpenAIStyle {
    Chat,
    Responses,
}

async fn run_openai_style(
    state: &AppStateData,
    headers: &HeaderMap,
    request: Value,
    style: OpenAIStyle,
    task_id: Uuid,
    api_key_public_id: Option<&str>,
    event_tx: mpsc::UnboundedSender<SerializedSseEvent>,
) -> Result<Value, AsyncInferenceError> {
    let mut openai_params: OpenAICompatibleParams = match style {
        OpenAIStyle::Chat => deserialize_stored_body(&request)?,
        OpenAIStyle::Responses => {
            let responses_params: OpenAICompatibleResponsesParams =
                deserialize_stored_body(&request)?;
            responses_params
                .into_chat_params()
                .map_err(|e| AsyncInferenceError::from_error(&e, false))?
        }
    };
    openai_params.stream = Some(true);

    let mut validated =
        validate_openai_compatible_request(headers, openai_params).map_err(|rejection| {
            AsyncInferenceError::from_error(&rejection.error, rejection.include_raw_response)
        })?;
    validated.params.include_aggregated_response = true;
    insert_api_key_public_id_tag(&mut validated.params.tags, api_key_public_id);
    insert_async_tags(&mut validated.params.tags, task_id);

    let include_usage = validated.include_usage;
    let include_raw_usage = validated.include_raw_usage;
    let include_original_response = validated.include_original_response;
    let include_raw_response = validated.include_raw_response;
    let response_model_prefix = validated.response_model_prefix.clone();
    let stream_aggregate = validated.synapse.stream_aggregate.clone();
    let anthropic_style = validated.synapse.response_style_anthropic;

    let inferred = execute_openai_compatible(state, None, validated)
        .await
        .map_err(|rejection| {
            AsyncInferenceError::from_error(&rejection.error, rejection.include_raw_response)
        })?;

    let InferenceOutput::Streaming(stream) = inferred.output else {
        return Err(AsyncInferenceError::internal(
            "Async inference expected a streaming inference output",
        ));
    };

    let (stream, sink) = tee_stream(stream, include_raw_response);
    let frames: Pin<Box<dyn Stream<Item = Result<SerializedSseEvent, Error>> + Send>> =
        match (style, anthropic_style) {
            (OpenAIStyle::Chat, true) => Box::pin(prepare_anthropic_sse(
                stream,
                response_model_prefix.clone(),
                stream_aggregate,
            )),
            (OpenAIStyle::Chat, false) => Box::pin(prepare_serialized_openai_compatible_events(
                stream,
                response_model_prefix.clone(),
                include_usage,
                include_raw_usage,
                include_original_response,
                include_raw_response,
                stream_aggregate,
            )),
            (OpenAIStyle::Responses, _) => Box::pin(prepare_serialized_openai_responses_events(
                stream,
                response_model_prefix.clone(),
            )),
        };

    drive_frames(frames, event_tx).await?;
    let response = finish_from_sink(sink)?;

    let shaped = match (style, anthropic_style) {
        (OpenAIStyle::Chat, true) => {
            let variant_name = response_variant_name(&response);
            serde_json::to_value(anthropic_from_inference(
                response,
                &format!("{response_model_prefix}{variant_name}"),
            ))
        }
        (OpenAIStyle::Chat, false) => serde_json::to_value(OpenAICompatibleResponse::from((
            response,
            response_model_prefix,
            include_original_response,
            include_raw_response,
        ))),
        (OpenAIStyle::Responses, _) => serde_json::to_value(
            OpenAICompatibleResponsesResponse::from((response, response_model_prefix)),
        ),
    };
    shaped.map_err(|e| {
        AsyncInferenceError::internal(format!(
            "Failed to serialize async inference final response: {e}"
        ))
    })
}

async fn run_messages_style(
    state: &AppStateData,
    headers: &HeaderMap,
    request: Value,
    task_id: Uuid,
    api_key_public_id: Option<&str>,
    event_tx: mpsc::UnboundedSender<SerializedSseEvent>,
) -> Result<Value, AsyncInferenceError> {
    let mut params: AnthropicMessagesParams = deserialize_stored_body(&request)?;
    params.stream = Some(true);

    let mut validated = validate_anthropic_request(headers, params)
        .map_err(|error| AsyncInferenceError::from_error(&error, false))?;
    validated.tz_params.include_aggregated_response = true;
    insert_api_key_public_id_tag(&mut validated.tz_params.tags, api_key_public_id);
    insert_async_tags(&mut validated.tz_params.tags, task_id);

    let response_model = validated.response_model.clone();
    let stream_aggregate = validated.synapse.stream_aggregate.clone();

    let (output, _synapse) = execute_anthropic(state, None, validated)
        .await
        .map_err(|rejection| AsyncInferenceError::from_error(&rejection.error, false))?;

    let InferenceOutput::Streaming(stream) = output else {
        return Err(AsyncInferenceError::internal(
            "Async inference expected a streaming inference output",
        ));
    };

    let (stream, sink) = tee_stream(stream, false);
    let frames = Box::pin(prepare_anthropic_sse(
        stream,
        response_model.clone(),
        stream_aggregate,
    ));
    drive_frames(frames, event_tx).await?;
    let response = finish_from_sink(sink)?;

    serde_json::to_value(anthropic_from_inference(response, &response_model)).map_err(|e| {
        AsyncInferenceError::internal(format!(
            "Failed to serialize async inference final response: {e}"
        ))
    })
}

fn deserialize_stored_body<T: DeserializeOwned>(body: &Value) -> Result<T, AsyncInferenceError> {
    serde_path_to_error::deserialize(body).map_err(|e| {
        AsyncInferenceError::internal(format!(
            "Failed to deserialize the stored async inference request: {e}"
        ))
    })
}

/// Captures from the raw inference stream: the last chunk (which carries the
/// aggregated response) and the first mid-stream error.
struct ChunkSink {
    last_chunk: Arc<Mutex<Option<InferenceResponseChunk>>>,
    stream_error: Arc<Mutex<Option<AsyncInferenceError>>>,
}

/// Wrap an [`InferenceStream`] so every chunk passing through is recorded in
/// the returned [`ChunkSink`], without changing what the serializer sees.
fn tee_stream(stream: InferenceStream, include_raw_response: bool) -> (InferenceStream, ChunkSink) {
    let sink = ChunkSink {
        last_chunk: Arc::new(Mutex::new(None)),
        stream_error: Arc::new(Mutex::new(None)),
    };
    let last_chunk = sink.last_chunk.clone();
    let stream_error = sink.stream_error.clone();
    let teed = stream.map(move |item| {
        match &item {
            Ok(chunk) => {
                if let Ok(mut guard) = last_chunk.lock() {
                    *guard = Some(chunk.clone());
                }
            }
            Err(error) => {
                if let Ok(mut guard) = stream_error.lock()
                    && guard.is_none()
                {
                    *guard = Some(AsyncInferenceError::from_error(error, include_raw_response));
                }
            }
        }
        item
    });
    (Box::pin(teed), sink)
}

/// Drive the serializer stream to completion, forwarding each frame to
/// `event_tx`. Serializer errors are collected and returned after the stream
/// ends (the stream is drained so model inferences and observability writes
/// finish).
async fn drive_frames(
    mut frames: Pin<Box<dyn Stream<Item = Result<SerializedSseEvent, Error>> + Send>>,
    event_tx: mpsc::UnboundedSender<SerializedSseEvent>,
) -> Result<(), AsyncInferenceError> {
    let mut first_error: Option<AsyncInferenceError> = None;
    while let Some(frame) = frames.next().await {
        match frame {
            Ok(frame) => {
                // If the receiver is gone (e.g. the task was cancelled), keep
                // draining the stream so the inference finishes cleanly.
                let _ = event_tx.send(frame);
            }
            Err(error) => {
                if first_error.is_none() {
                    first_error = Some(AsyncInferenceError::from_error(&error, false));
                }
            }
        }
    }
    match first_error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// Build the final [`InferenceResponse`] from the captured stream state.
fn finish_from_sink(sink: ChunkSink) -> Result<InferenceResponse, AsyncInferenceError> {
    if let Ok(mut guard) = sink.stream_error.lock()
        && let Some(error) = guard.take()
    {
        return Err(error);
    }
    let last_chunk = sink
        .last_chunk
        .lock()
        .ok()
        .and_then(|mut guard| guard.take());
    let Some(chunk) = last_chunk else {
        return Err(AsyncInferenceError::internal(
            "Async inference stream ended without any chunks",
        ));
    };
    match chunk {
        InferenceResponseChunk::Chat(chunk) => {
            let ChatInferenceResponseChunk {
                inference_id,
                episode_id,
                variant_name,
                usage,
                raw_usage,
                raw_response,
                finish_reason,
                aggregated_response,
                ..
            } = chunk;
            let Some(content) = aggregated_response else {
                return Err(AsyncInferenceError::internal(
                    "Async inference stream finished without an aggregated response",
                ));
            };
            Ok(InferenceResponse::Chat(ChatInferenceResponse {
                inference_id,
                episode_id,
                variant_name,
                content,
                usage: usage.unwrap_or_default(),
                raw_usage,
                original_response: None,
                raw_response,
                finish_reason,
            }))
        }
        InferenceResponseChunk::Json(chunk) => {
            let JsonInferenceResponseChunk {
                inference_id,
                episode_id,
                variant_name,
                usage,
                raw_usage,
                raw_response,
                finish_reason,
                aggregated_response,
                ..
            } = chunk;
            let Some(output) = aggregated_response else {
                return Err(AsyncInferenceError::internal(
                    "Async inference stream finished without an aggregated response",
                ));
            };
            Ok(InferenceResponse::Json(JsonInferenceResponse {
                inference_id,
                episode_id,
                variant_name,
                output,
                usage: usage.unwrap_or_default(),
                raw_usage,
                original_response: None,
                raw_response,
                finish_reason,
            }))
        }
    }
}

fn response_variant_name(response: &InferenceResponse) -> String {
    match response {
        InferenceResponse::Chat(chat) => chat.variant_name.clone(),
        InferenceResponse::Json(json_response) => json_response.variant_name.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use googletest::prelude::*;
    use std::collections::HashMap;
    use tensorzero_auth::key::TensorZeroApiKey;
    use tensorzero_auth::postgres::KeyInfo;

    fn header_map(entries: &[(&str, &str)]) -> HeaderMap {
        let mut headers = HeaderMap::new();
        for (name, value) in entries {
            headers.insert(
                name.parse::<HeaderName>().expect("valid header name"),
                HeaderValue::from_str(value).expect("valid header value"),
            );
        }
        headers
    }

    #[gtest]
    fn capture_headers_keeps_compat_headers_and_drops_credentials() {
        let headers = header_map(&[
            ("authorization", "Bearer secret"),
            ("x-synapse-timeout-ms", "5000"),
            ("x-tensorzero-tags", "a=b"),
            ("x-request-id", "req-42"),
            ("content-type", "application/json"),
            ("x-api-key", "secret"),
        ]);
        let captured = capture_headers(&headers);
        expect_eq!(captured.len(), 3);
        expect_that!(
            captured.get("x-synapse-timeout-ms").map(String::as_str),
            some(eq("5000"))
        );
        expect_that!(
            captured.get("x-tensorzero-tags").map(String::as_str),
            some(eq("a=b"))
        );
        expect_that!(
            captured.get("x-request-id").map(String::as_str),
            some(eq("req-42"))
        );
        expect_that!(captured.get("authorization"), none());
        expect_that!(captured.get("x-api-key"), none());
        expect_that!(captured.get("content-type"), none());
    }

    #[gtest]
    fn rebuild_headers_roundtrips_captured_headers() {
        let headers = header_map(&[("x-request-id", "req-42"), ("x-synapse-timeout-ms", "5000")]);
        let captured = capture_headers(&headers);
        let rebuilt = rebuild_headers(&captured).expect("should rebuild");
        expect_that!(
            rebuilt.get("x-request-id").and_then(|v| v.to_str().ok()),
            some(eq("req-42"))
        );
        expect_that!(
            rebuilt
                .get("x-synapse-timeout-ms")
                .and_then(|v| v.to_str().ok()),
            some(eq("5000"))
        );
    }

    fn stream_entry(fields: &[(&str, &str)]) -> StreamId {
        StreamId {
            id: "1-0".to_string(),
            map: fields
                .iter()
                .map(|(k, v)| {
                    (
                        (*k).to_string(),
                        redis::Value::BulkString(v.as_bytes().to_vec()),
                    )
                })
                .collect::<HashMap<String, redis::Value>>(),
            milliseconds_elapsed_from_delivery: None,
            delivered_count: None,
        }
    }

    #[gtest]
    fn parse_stream_entry_plain_event() {
        let entry = stream_entry(&[
            (STREAM_FIELD_EVENT, "content_block_delta"),
            (STREAM_FIELD_DATA, "{\"delta\":1}"),
        ]);
        let action = parse_stream_entry(&entry).expect("should parse");
        let StreamEntryAction::Event(event) = action else {
            panic!("expected a plain event");
        };
        // axum `Event` has no getters; the debug format includes the fields.
        let debug = format!("{event:?}");
        expect_that!(debug, contains_substring("content_block_delta"));
        expect_that!(debug, contains_substring("delta"));
    }

    #[gtest]
    fn parse_stream_entry_done_marker_ends_stream() {
        let entry = stream_entry(&[(STREAM_FIELD_MARKER, STREAM_MARKER_DONE)]);
        let action = parse_stream_entry(&entry).expect("should parse");
        assert_that!(action, matches_pattern!(StreamEntryAction::End));
    }

    #[gtest]
    fn parse_stream_entry_error_marker_is_terminal_event() {
        let entry = stream_entry(&[
            (STREAM_FIELD_MARKER, STREAM_MARKER_ERROR),
            (STREAM_FIELD_DATA, "{\"error\":{\"message\":\"boom\"}}"),
        ]);
        let action = parse_stream_entry(&entry).expect("should parse");
        let StreamEntryAction::TerminalEvent(event) = action else {
            panic!("expected a terminal event");
        };
        let debug = format!("{event:?}");
        expect_that!(debug, contains_substring("boom"));
    }

    #[gtest]
    fn parse_stream_entry_missing_data_fails() {
        let entry = stream_entry(&[(STREAM_FIELD_EVENT, "chunk")]);
        expect_that!(parse_stream_entry(&entry), err(anything()));
    }

    #[gtest]
    fn deserialize_body_reports_serde_path() {
        let body = json!({"model": 42, "messages": []});
        let error = deserialize_body::<OpenAICompatibleParams>(&body)
            .expect_err("invalid model type should fail");
        expect_that!(error.to_string(), contains_substring("model"));
    }

    fn test_api_key_ext() -> Extension<RequestApiKeyExtension> {
        let key = TensorZeroApiKey::parse(
            "sk-t0-abcdefghijkl-123456789012345678901234567890123456789012345678",
        )
        .expect("valid test API key");
        Extension(RequestApiKeyExtension {
            api_key: Arc::new(key),
            key_info: KeyInfo {
                public_id: "abcdefghijkl".to_string(),
                organization: "test-org".to_string(),
                workspace: "test-workspace".to_string(),
                description: None,
                created_at: Utc::now(),
                disabled_at: None,
                expires_at: None,
            },
        })
    }

    #[gtest]
    fn submit_params_carry_api_key_public_id_when_authenticated() {
        let ext = test_api_key_ext();
        let public_id = api_key_public_id(Some(&ext));
        expect_that!(public_id.as_deref(), some(eq("abcdefghijkl")));

        let params = AsyncInferenceTaskParams {
            api_kind: AsyncInferenceApiKind::Chat,
            request: json!({"model": "openai::gpt-5", "messages": []}),
            headers: BTreeMap::new(),
            api_key_public_id: public_id,
        };
        let value = serde_json::to_value(&params).expect("should serialize");
        expect_that!(&value["api_key_public_id"], eq(&json!("abcdefghijkl")));
    }

    #[gtest]
    fn submit_params_omit_api_key_public_id_when_unauthenticated() {
        expect_that!(api_key_public_id(None), none());

        let params = AsyncInferenceTaskParams {
            api_kind: AsyncInferenceApiKind::Chat,
            request: json!({"model": "openai::gpt-5", "messages": []}),
            headers: BTreeMap::new(),
            api_key_public_id: api_key_public_id(None),
        };
        let value = serde_json::to_value(&params).expect("should serialize");
        expect_that!(value.get("api_key_public_id"), none());
    }

    #[gtest]
    fn worker_path_inserts_api_key_public_id_tag_when_present() {
        let mut tags = HashMap::new();
        insert_api_key_public_id_tag(&mut tags, Some("abcdefghijkl"));
        expect_that!(
            tags.get(API_KEY_PUBLIC_ID_TAG).map(String::as_str),
            some(eq("abcdefghijkl"))
        );
    }

    #[gtest]
    fn worker_path_leaves_tags_untouched_without_api_key() {
        let mut tags = HashMap::new();
        insert_api_key_public_id_tag(&mut tags, None);
        expect_that!(tags.contains_key(API_KEY_PUBLIC_ID_TAG), eq(false));
    }

    #[gtest]
    fn worker_path_marks_inference_async_with_task_id() {
        let task_id = Uuid::now_v7();
        let expected_task_id = task_id.to_string();
        let mut tags = HashMap::new();
        insert_async_tags(&mut tags, task_id);
        expect_that!(tags.get(ASYNC_TAG).map(String::as_str), some(eq("true")));
        expect_that!(
            tags.get(ASYNC_TASK_ID_TAG).map(String::as_str),
            some(eq(expected_task_id.as_str()))
        );
    }

    #[gtest]
    fn openai_style_worker_tags_override_body_tags() {
        let task_id = Uuid::now_v7();
        let expected_task_id = task_id.to_string();
        let body = json!({
            "model": "openai::gpt-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tensorzero::tags": {
                "tensorzero::async": "false",
                "tensorzero::async_task_id": "caller-supplied",
                "caller_tag": "kept"
            }
        });
        let params: OpenAICompatibleParams =
            serde_json::from_value(body).expect("should deserialize");
        let mut validated = validate_openai_compatible_request(&HeaderMap::new(), params)
            .unwrap_or_else(|rejection| panic!("should validate: {}", rejection.error));

        // Same stamping sequence as `run_openai_style`, after validation.
        insert_api_key_public_id_tag(&mut validated.params.tags, None);
        insert_async_tags(&mut validated.params.tags, task_id);

        expect_that!(
            validated.params.tags.get(ASYNC_TAG).map(String::as_str),
            some(eq("true"))
        );
        expect_that!(
            validated
                .params
                .tags
                .get(ASYNC_TASK_ID_TAG)
                .map(String::as_str),
            some(eq(expected_task_id.as_str()))
        );
        expect_that!(
            validated.params.tags.get("caller_tag").map(String::as_str),
            some(eq("kept"))
        );
    }

    #[gtest]
    fn messages_style_worker_tags_mark_task_id() {
        let task_id = Uuid::now_v7();
        let expected_task_id = task_id.to_string();
        let headers = header_map(&[("x-tensorzero-tags", "env=prod")]);
        let body = json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}]
        });
        let params: AnthropicMessagesParams =
            serde_json::from_value(body).expect("should deserialize");
        let mut validated = validate_anthropic_request(&headers, params).expect("should validate");

        // Same stamping sequence as `run_messages_style`, after validation.
        insert_api_key_public_id_tag(&mut validated.tz_params.tags, None);
        insert_async_tags(&mut validated.tz_params.tags, task_id);

        expect_that!(
            validated.tz_params.tags.get(ASYNC_TAG).map(String::as_str),
            some(eq("true"))
        );
        expect_that!(
            validated
                .tz_params
                .tags
                .get(ASYNC_TASK_ID_TAG)
                .map(String::as_str),
            some(eq(expected_task_id.as_str()))
        );
        expect_that!(
            validated.tz_params.tags.get("env").map(String::as_str),
            some(eq("prod"))
        );
    }
}

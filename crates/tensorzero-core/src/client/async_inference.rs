// Modified by Delta-AI under Apache 2.0
//! Client methods for the async inference API (`POST .../async` submit
//! endpoints, `GET /v1/async_tasks/{task_id}` status polling,
//! `GET /v1/async_tasks/{task_id}/stream` SSE follow) and the unauthenticated
//! `GET /status` / `GET /health` gateway endpoints.
//!
//! All methods in this module are HTTP-only: the async inference API requires
//! the gateway's durable task queue (Postgres) and event stream (Valkey), so
//! embedded gateway mode returns an explicit error, matching the convention of
//! `Client::http_inference` and other HTTP-only client methods.

use std::pin::Pin;
use std::time::Duration;

use futures::Stream;
use serde_json::Value;
use tokio::time::Instant;
use tokio_stream::StreamExt;
use uuid::Uuid;

use crate::endpoints::openai_compatible::async_inference_types::{
    AsyncInferenceApiKind, AsyncInferenceLaunchResponse, AsyncTaskStatusResponse,
};
use crate::endpoints::status::StatusResponse;
use crate::error::{Error, ErrorDetails};

use super::{Client, ClientMode, DisplayOrDebug, HTTPGateway, TensorZeroError};

/// One SSE event from `GET /v1/async_tasks/{task_id}/stream`.
///
/// The `data` payload is the raw JSON string in the wire shape of the API the
/// task was submitted to (OpenAI chat completions chunks, OpenAI responses
/// events, or Anthropic messages events). The `event` field carries the SSE
/// `event:` name when the event has one (e.g. named responses-API events).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct AsyncTaskStreamEvent {
    /// The SSE `event:` field (`None` = unnamed event, the SSE default).
    pub event: Option<String>,
    /// The SSE `data:` field.
    pub data: String,
}

/// Stream of SSE events for an async inference task, replaying events written
/// so far and then following the task live. The stream ends when the task
/// reaches a terminal state; a terminal `error` marker is surfaced as a
/// `TensorZeroError` item carrying the error body in `raw_event`.
pub type AsyncTaskEventStream =
    Pin<Box<dyn Stream<Item = Result<AsyncTaskStreamEvent, TensorZeroError>> + Send>>;

/// Options for [`Client::wait_for_async_task`] polling.
#[derive(Clone, Copy, Debug)]
pub struct AsyncTaskWaitOptions {
    /// Delay before the first re-poll after a non-terminal status.
    pub initial_interval: Duration,
    /// Upper bound for the exponential backoff between polls.
    pub max_interval: Duration,
    /// Total time budget; the call errors if the task is not terminal by then.
    pub timeout: Duration,
}

impl Default for AsyncTaskWaitOptions {
    fn default() -> Self {
        Self {
            initial_interval: Duration::from_millis(500),
            max_interval: Duration::from_secs(5),
            timeout: Duration::from_secs(300),
        }
    }
}

/// The URL path of the submit endpoint for each async inference API kind.
fn submit_path(kind: AsyncInferenceApiKind) -> &'static str {
    match kind {
        AsyncInferenceApiKind::Chat => "v1/chat/completions/async",
        AsyncInferenceApiKind::Responses => "v1/responses/async",
        AsyncInferenceApiKind::Messages => "v1/messages/async",
    }
}

fn is_terminal(status: &AsyncTaskStatusResponse) -> bool {
    matches!(
        status,
        AsyncTaskStatusResponse::Completed { .. }
            | AsyncTaskStatusResponse::Failed { .. }
            | AsyncTaskStatusResponse::Cancelled { .. }
    )
}

fn join_url(client: &HTTPGateway, endpoint: &str) -> Result<reqwest::Url, TensorZeroError> {
    client
        .base_url
        .join(endpoint)
        .map_err(|e| TensorZeroError::Other {
            source: Error::new(ErrorDetails::InvalidBaseUrl {
                message: format!("Failed to join base URL with /{endpoint} endpoint: {e}"),
            })
            .into(),
        })
}

fn http_only_error(method: &str) -> TensorZeroError {
    TensorZeroError::Other {
        source: Error::new(ErrorDetails::InternalError {
            message: format!(
                "`{method}` is not supported in embedded gateway mode; it requires the async inference HTTP API of a running gateway"
            ),
        })
        .into(),
    }
}

impl Client {
    /// Submits an async inference job to the gateway
    /// (`POST /v1/chat/completions/async`, `/v1/responses/async`, or
    /// `/v1/messages/async`, selected by `kind`).
    ///
    /// `request` is the raw JSON body of the corresponding synchronous API;
    /// its `stream` field is ignored by the gateway. The gateway validates the
    /// body synchronously and returns `202 Accepted` with the durable task ID.
    ///
    /// Only available in `HTTPGateway` mode.
    pub async fn submit_async_inference(
        &self,
        kind: AsyncInferenceApiKind,
        request: Value,
    ) -> Result<AsyncInferenceLaunchResponse, TensorZeroError> {
        let ClientMode::HTTPGateway(client) = &*self.mode else {
            return Err(http_only_error("submit_async_inference"));
        };
        let path = submit_path(kind);
        let url = join_url(client, path)?;
        let builder = client.http_client.post(url).json(&request);
        Ok(client.send_and_parse_http_response(builder).await?.0)
    }

    /// Fetches the current status of an async inference task
    /// (`GET /v1/async_tasks/{task_id}`).
    ///
    /// A missing task surfaces as `TensorZeroError::Http` with status 404; a
    /// gateway without async inference enabled surfaces status 500.
    ///
    /// Only available in `HTTPGateway` mode.
    pub async fn get_async_task(
        &self,
        task_id: Uuid,
    ) -> Result<AsyncTaskStatusResponse, TensorZeroError> {
        let ClientMode::HTTPGateway(client) = &*self.mode else {
            return Err(http_only_error("get_async_task"));
        };
        let url = join_url(client, &format!("v1/async_tasks/{task_id}"))?;
        let builder = client.http_client.get(url);
        Ok(client.send_and_parse_http_response(builder).await?.0)
    }

    /// Attaches to the SSE event stream of an async inference task
    /// (`GET /v1/async_tasks/{task_id}/stream`), replaying events written so
    /// far and then following the task live until it terminates.
    ///
    /// Each yielded event's `data` is the raw payload in the wire shape of the
    /// API the task was submitted to. The gateway ends every stream with an
    /// explicit terminal frame: `data: [DONE]` on success, which is swallowed
    /// here (the stream simply ends), matching the behavior of the synchronous
    /// streaming client; or `event: error`, which is yielded as an error item
    /// carrying the error body.
    ///
    /// If the task already finished and its event stream has expired, the
    /// gateway answers 410 and this returns a `TensorZeroError::Http`; fetch
    /// the final result with [`Client::get_async_task`] instead.
    ///
    /// Only available in `HTTPGateway` mode.
    pub async fn stream_async_task(
        &self,
        task_id: Uuid,
    ) -> Result<AsyncTaskEventStream, TensorZeroError> {
        let ClientMode::HTTPGateway(client) = &*self.mode else {
            return Err(http_only_error("stream_async_task"));
        };
        let url = join_url(client, &format!("v1/async_tasks/{task_id}/stream"))?;
        let builder = client.http_client.get(url);
        // Use the no-timeout builder: async task streams are long-running and
        // should not be subject to the per-request HTTP timeout.
        let event_source = match client
            .customize_builder_no_timeout(builder)
            .eventsource()
            .await
        {
            Ok(es) => es,
            Err(e) => {
                let inner_err = Error::new(ErrorDetails::StreamError {
                    source: Box::new(Error::new(ErrorDetails::Serialization {
                        message: format!("Error opening async task event stream: {e:?}"),
                    })),
                    raw_event: None,
                });
                if let reqwest_sse_stream::ReqwestSseStreamError::InvalidStatusCode(code, resp) = e
                {
                    return Err(TensorZeroError::Http {
                        status_code: code.as_u16(),
                        text: resp.text().await.ok(),
                        source: inner_err.into(),
                    });
                }
                return Err(TensorZeroError::Other {
                    source: inner_err.into(),
                });
            }
        };

        // `reqwest-sse-stream` delivers handshake failures (e.g. a 404 for an
        // unknown task or a 410 for an expired stream) as the first stream
        // item rather than failing `eventsource()`, so peek at the first item
        // and surface it as a proper error, mirroring
        // `HTTPGateway::send_http_stream_inference`.
        let mut event_source = Box::pin(event_source.peekable());
        if let Some(Err(_)) = event_source.peek().await {
            let res = event_source.next().await;
            let Some(Err(e)) = res else {
                // Unreachable: we just peeked an error on this stream.
                return Err(TensorZeroError::Other {
                    source: Error::new(ErrorDetails::InternalError {
                        message: "Async task event stream changed between peek and read"
                            .to_string(),
                    })
                    .into(),
                });
            };
            let inner_err = Error::new(ErrorDetails::StreamError {
                source: Box::new(Error::new(ErrorDetails::Serialization {
                    message: format!("Error opening async task event stream: {e:?}"),
                })),
                raw_event: None,
            });
            if let reqwest_sse_stream::ReqwestSseStreamError::InvalidStatusCode(code, resp) = *e {
                return Err(TensorZeroError::Http {
                    status_code: code.as_u16(),
                    text: resp.text().await.ok(),
                    source: inner_err.into(),
                });
            }
            return Err(TensorZeroError::Other {
                source: inner_err.into(),
            });
        }

        let verbose_errors = self.verbose_errors;
        Ok(Box::pin(async_stream::stream! {
            while let Some(ev) = event_source.next().await {
                match ev {
                    Err(e) => {
                        yield Err(TensorZeroError::Other {
                            source: Error::new(ErrorDetails::StreamError {
                                source: Box::new(Error::new(ErrorDetails::Serialization {
                                    message: format!("Error in async task event stream: {}", DisplayOrDebug {
                                        val: e,
                                        debug: verbose_errors,
                                    }),
                                })),
                                raw_event: None,
                            })
                            .into(),
                        });
                    }
                    Ok(reqwest_sse_stream::Event::Open) => continue,
                    Ok(reqwest_sse_stream::Event::Message(message)) => {
                        if message.data == "[DONE]" {
                            break;
                        }
                        if message.event == "error" {
                            let data = message.data;
                            let inner_err = Error::new(ErrorDetails::Serialization {
                                message: format!("Async task event stream produced an error: {}", DisplayOrDebug {
                                    val: &data,
                                    debug: verbose_errors,
                                }),
                            });
                            yield Err(TensorZeroError::Other {
                                source: Error::new(ErrorDetails::StreamError {
                                    source: Box::new(inner_err),
                                    raw_event: Some(data),
                                })
                                .into(),
                            });
                            break;
                        }
                        let event = if message.event.is_empty() {
                            None
                        } else {
                            Some(message.event)
                        };
                        yield Ok(AsyncTaskStreamEvent {
                            event,
                            data: message.data,
                        });
                    }
                }
            }
        }))
    }

    /// Polls [`Client::get_async_task`] with exponential backoff until the task
    /// reaches a terminal state (completed / failed / cancelled), and returns
    /// the terminal status.
    ///
    /// Errors when the task is not terminal within `options.timeout`, or when a
    /// status poll itself fails.
    ///
    /// Only available in `HTTPGateway` mode.
    pub async fn wait_for_async_task(
        &self,
        task_id: Uuid,
        options: AsyncTaskWaitOptions,
    ) -> Result<AsyncTaskStatusResponse, TensorZeroError> {
        let start = Instant::now();
        let mut interval = options.initial_interval;
        loop {
            let status = self.get_async_task(task_id).await?;
            if is_terminal(&status) {
                return Ok(status);
            }
            if start.elapsed() + interval > options.timeout {
                return Err(TensorZeroError::Other {
                    source: Error::new(ErrorDetails::InternalError {
                        message: format!(
                            "Async inference task `{task_id}` did not reach a terminal state within {:?}",
                            options.timeout
                        ),
                    })
                    .into(),
                });
            }
            tokio::time::sleep(interval).await;
            interval = (interval * 2).min(options.max_interval);
        }
    }

    /// Fetches the gateway's liveness status (`GET /status`, unauthenticated).
    ///
    /// Only available in `HTTPGateway` mode.
    pub async fn status(&self) -> Result<StatusResponse, TensorZeroError> {
        let ClientMode::HTTPGateway(client) = &*self.mode else {
            return Err(http_only_error("status"));
        };
        let url = join_url(client, "status")?;
        let builder = client.http_client.get(url);
        Ok(client.send_and_parse_http_response(builder).await?.0)
    }

    /// Fetches the gateway's health report (`GET /health`, unauthenticated),
    /// covering the gateway itself and its ClickHouse / Postgres / Valkey
    /// dependencies.
    ///
    /// When any dependency is unhealthy the gateway answers 503; this surfaces
    /// as a `TensorZeroError::Http` whose `text` carries the per-service
    /// health JSON body.
    ///
    /// Only available in `HTTPGateway` mode.
    pub async fn health(&self) -> Result<Value, TensorZeroError> {
        let ClientMode::HTTPGateway(client) = &*self.mode else {
            return Err(http_only_error("health"));
        };
        let url = join_url(client, "health")?;
        let builder = client.http_client.get(url);
        Ok(client.send_and_parse_http_response(builder).await?.0)
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use axum::extract::{OriginalUri, State};
    use axum::http::StatusCode;
    use axum::response::sse::{Event, Sse};
    use axum::routing::{get, post};
    use axum::{Json, Router};
    use futures::stream;
    use googletest::prelude::*;
    use serde_json::json;
    use url::Url;

    use crate::client::{ClientBuilder, ClientBuilderMode};

    use super::*;

    const TEST_TASK_ID: &str = "0190f9c4-8e3a-7b3d-9c1e-2f4a5b6c7d8e";

    fn test_task_id() -> Uuid {
        Uuid::parse_str(TEST_TASK_ID).expect("valid UUID")
    }

    /// Starts the router on a loopback port and returns an HTTP-mode client
    /// pointed at it.
    #[expect(clippy::disallowed_methods)]
    async fn spawn_mock_gateway(router: Router) -> Client {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("mock gateway should bind");
        let addr = listener
            .local_addr()
            .expect("mock gateway should have a local address");
        tokio::spawn(async move {
            axum::serve(listener, router)
                .await
                .expect("mock gateway should serve");
        });
        ClientBuilder::new(ClientBuilderMode::HTTPGateway {
            url: Url::parse(&format!("http://{addr}/")).expect("mock gateway URL should parse"),
        })
        .build_http()
        .expect("HTTP client should build")
    }

    // ------------------------------------------------------------------
    // submit_async_inference
    // ------------------------------------------------------------------

    #[derive(Clone, Default)]
    struct SubmitRecorder {
        requests: Arc<Mutex<Vec<(String, Value)>>>,
    }

    async fn record_submit(
        State(recorder): State<SubmitRecorder>,
        OriginalUri(uri): OriginalUri,
        Json(body): Json<Value>,
    ) -> (StatusCode, Json<Value>) {
        recorder
            .requests
            .lock()
            .expect("recorder lock should not be poisoned")
            .push((uri.path().to_string(), body));
        (StatusCode::ACCEPTED, Json(json!({"task_id": TEST_TASK_ID})))
    }

    #[gtest]
    #[tokio::test]
    async fn submit_async_inference_posts_to_kind_path() {
        let recorder = SubmitRecorder::default();
        let router = Router::new()
            .route("/v1/chat/completions/async", post(record_submit))
            .route("/v1/responses/async", post(record_submit))
            .route("/v1/messages/async", post(record_submit))
            .with_state(recorder.clone());
        let client = spawn_mock_gateway(router).await;

        for (kind, expected_path) in [
            (AsyncInferenceApiKind::Chat, "/v1/chat/completions/async"),
            (AsyncInferenceApiKind::Responses, "/v1/responses/async"),
            (AsyncInferenceApiKind::Messages, "/v1/messages/async"),
        ] {
            let body = json!({"model": "test-model"});
            let launch = client
                .submit_async_inference(kind, body)
                .await
                .expect("submit should succeed");
            expect_that!(launch.task_id, eq(test_task_id()), "path: {expected_path}");
        }

        let requests = recorder
            .requests
            .lock()
            .expect("recorder lock should not be poisoned");
        let paths: Vec<&str> = requests.iter().map(|(path, _)| path.as_str()).collect();
        expect_that!(
            &paths,
            eq(&vec![
                "/v1/chat/completions/async",
                "/v1/responses/async",
                "/v1/messages/async"
            ])
        );
        expect_that!(requests.len(), eq(3));
    }

    // ------------------------------------------------------------------
    // get_async_task
    // ------------------------------------------------------------------

    /// Serves `body` with `status` for every `GET /v1/async_tasks/{task_id}`.
    fn status_router(status: StatusCode, body: Value) -> Router {
        Router::new().route(
            "/v1/async_tasks/{task_id}",
            get(move || {
                let body = body.clone();
                async move { (status, Json(body)) }
            }),
        )
    }

    #[gtest]
    #[tokio::test]
    async fn get_async_task_parses_each_status_variant() {
        for (body, expected) in [
            (
                json!({"status": "queued", "task_id": TEST_TASK_ID, "queue_position": 2}),
                AsyncTaskStatusResponse::Queued {
                    task_id: test_task_id(),
                    queue_position: Some(2),
                },
            ),
            (
                json!({"status": "running", "task_id": TEST_TASK_ID, "elapsed_ms": 42}),
                AsyncTaskStatusResponse::Running {
                    task_id: test_task_id(),
                    started_at: None,
                    elapsed_ms: Some(42),
                },
            ),
            (
                json!({"status": "completed", "task_id": TEST_TASK_ID, "response": {"id": "chatcmpl-1"}}),
                AsyncTaskStatusResponse::Completed {
                    task_id: test_task_id(),
                    response: json!({"id": "chatcmpl-1"}),
                },
            ),
            (
                json!({"status": "failed", "task_id": TEST_TASK_ID, "error": {"message": "boom"}}),
                AsyncTaskStatusResponse::Failed {
                    task_id: test_task_id(),
                    error: Some(json!({"message": "boom"})),
                },
            ),
            (
                json!({"status": "cancelled", "task_id": TEST_TASK_ID}),
                AsyncTaskStatusResponse::Cancelled {
                    task_id: test_task_id(),
                    error: None,
                },
            ),
        ] {
            let client = spawn_mock_gateway(status_router(StatusCode::OK, body)).await;
            let status = client
                .get_async_task(test_task_id())
                .await
                .expect("status fetch should succeed");
            expect_that!(&status, eq(&expected));
        }
    }

    #[gtest]
    #[tokio::test]
    async fn get_async_task_unknown_id_surfaces_404() {
        let client = spawn_mock_gateway(status_router(
            StatusCode::NOT_FOUND,
            json!({"error": "Route not found"}),
        ))
        .await;
        let err = client
            .get_async_task(test_task_id())
            .await
            .expect_err("unknown task should error");
        match err {
            TensorZeroError::Http { status_code, .. } => expect_that!(status_code, eq(404)),
            other => panic!("expected a 404 HTTP error, got {other:?}"),
        }
    }

    // ------------------------------------------------------------------
    // wait_for_async_task
    // ------------------------------------------------------------------

    struct PollSequence {
        calls: AtomicUsize,
        responses: Vec<Value>,
    }

    impl PollSequence {
        /// Replies with `responses[call_index]`, repeating the last response
        /// once the sequence is exhausted.
        fn router(responses: Vec<Value>) -> (Router, Arc<Self>) {
            let state = Arc::new(Self {
                calls: AtomicUsize::new(0),
                responses,
            });
            let router = Router::new()
                .route("/v1/async_tasks/{task_id}", get(poll_sequence_handler))
                .with_state(state.clone());
            (router, state)
        }
    }

    async fn poll_sequence_handler(State(state): State<Arc<PollSequence>>) -> Json<Value> {
        let call = state.calls.fetch_add(1, Ordering::SeqCst);
        let index = call.min(state.responses.len() - 1);
        Json(state.responses[index].clone())
    }

    fn fast_options() -> AsyncTaskWaitOptions {
        AsyncTaskWaitOptions {
            initial_interval: Duration::from_millis(10),
            max_interval: Duration::from_millis(20),
            timeout: Duration::from_secs(10),
        }
    }

    #[gtest]
    #[tokio::test]
    async fn wait_for_async_task_polls_until_terminal() {
        let (router, state) = PollSequence::router(vec![
            json!({"status": "queued", "task_id": TEST_TASK_ID, "queue_position": 1}),
            json!({"status": "running", "task_id": TEST_TASK_ID}),
            json!({"status": "completed", "task_id": TEST_TASK_ID, "response": {"id": "chatcmpl-1"}}),
        ]);
        let client = spawn_mock_gateway(router).await;

        let status = client
            .wait_for_async_task(test_task_id(), fast_options())
            .await
            .expect("task should complete");
        expect_that!(
            &status,
            eq(&AsyncTaskStatusResponse::Completed {
                task_id: test_task_id(),
                response: json!({"id": "chatcmpl-1"}),
            })
        );
        expect_that!(
            state.calls.load(Ordering::SeqCst),
            eq(3),
            "wait should poll until the terminal status"
        );
    }

    #[gtest]
    #[tokio::test]
    async fn wait_for_async_task_times_out_on_non_terminal_task() {
        let (router, _state) =
            PollSequence::router(vec![json!({"status": "queued", "task_id": TEST_TASK_ID})]);
        let client = spawn_mock_gateway(router).await;

        let err = client
            .wait_for_async_task(
                test_task_id(),
                AsyncTaskWaitOptions {
                    timeout: Duration::from_millis(50),
                    ..fast_options()
                },
            )
            .await
            .expect_err("a task that never terminates should time out");
        expect_that!(
            err.to_string(),
            contains_substring("did not reach a terminal state")
        );
    }

    // ------------------------------------------------------------------
    // stream_async_task
    // ------------------------------------------------------------------

    fn sse_router(events: Vec<Event>) -> Router {
        let events = Arc::new(events);
        Router::new().route(
            "/v1/async_tasks/{task_id}/stream",
            get(move || {
                let events = events.clone();
                async move {
                    Sse::new(stream::iter(
                        events
                            .iter()
                            .cloned()
                            .map(Ok::<Event, Infallible>)
                            .collect::<Vec<_>>(),
                    ))
                }
            }),
        )
    }

    #[gtest]
    #[tokio::test]
    async fn stream_async_task_yields_events_until_done_sentinel() {
        let client = spawn_mock_gateway(sse_router(vec![
            Event::default().data(r#"{"object":"chat.completion.chunk"}"#),
            Event::default()
                .event("response.created")
                .data(r#"{"type":"response.created"}"#),
            Event::default().data("[DONE]"),
        ]))
        .await;

        let events: Vec<_> = client
            .stream_async_task(test_task_id())
            .await
            .expect("stream should open")
            .collect()
            .await;

        expect_that!(events.len(), eq(2), "[DONE] should end the stream");
        expect_that!(
            &events[0],
            ok(eq(&AsyncTaskStreamEvent {
                event: None,
                data: r#"{"object":"chat.completion.chunk"}"#.to_string(),
            }))
        );
        expect_that!(
            &events[1],
            ok(eq(&AsyncTaskStreamEvent {
                event: Some("response.created".to_string()),
                data: r#"{"type":"response.created"}"#.to_string(),
            }))
        );
    }

    #[gtest]
    #[tokio::test]
    async fn stream_async_task_surfaces_terminal_error_event() {
        let client = spawn_mock_gateway(sse_router(vec![
            Event::default().data(r#"{"object":"chat.completion.chunk"}"#),
            Event::default()
                .event("error")
                .data(r#"{"error":{"message":"boom"}}"#),
        ]))
        .await;

        let events: Vec<_> = client
            .stream_async_task(test_task_id())
            .await
            .expect("stream should open")
            .collect()
            .await;

        expect_that!(events.len(), eq(2), "error marker should end the stream");
        expect_that!(&events[0], ok(anything()));
        match &events[1] {
            Err(TensorZeroError::Other { source }) => {
                expect_that!(source.to_string(), contains_substring("produced an error"));
            }
            other => panic!("expected an error item for the terminal error marker, got {other:?}"),
        }
    }

    #[gtest]
    #[tokio::test]
    async fn stream_async_task_expired_stream_surfaces_410() {
        let router = Router::new().route(
            "/v1/async_tasks/{task_id}/stream",
            get(|| async {
                (
                    StatusCode::GONE,
                    Json(json!({"error": {"message": "stream gone"}})),
                )
            }),
        );
        let client = spawn_mock_gateway(router).await;

        let err = match client.stream_async_task(test_task_id()).await {
            Err(err) => err,
            Ok(_) => panic!("an expired stream should error"),
        };
        match err {
            TensorZeroError::Http { status_code, .. } => expect_that!(status_code, eq(410)),
            other => panic!("expected a 410 HTTP error, got {other:?}"),
        }
    }

    // ------------------------------------------------------------------
    // status / health
    // ------------------------------------------------------------------

    #[gtest]
    #[tokio::test]
    async fn status_returns_gateway_status() {
        let router = Router::new().route(
            "/status",
            get(|| async {
                Json(json!({
                    "status": "ok",
                    "version": "2026.9.0",
                    "config_hash": "abcd1234",
                }))
            }),
        );
        let client = spawn_mock_gateway(router).await;

        let status = client.status().await.expect("status should succeed");
        expect_that!(status.status, eq("ok"));
        expect_that!(status.version, eq("2026.9.0"));
        expect_that!(status.config_hash, eq("abcd1234"));
    }

    #[gtest]
    #[tokio::test]
    async fn health_returns_service_report() {
        let router = Router::new().route(
            "/health",
            get(|| async {
                Json(json!({
                    "gateway": "ok",
                    "clickhouse": "ok",
                    "postgres": "ok",
                    "valkey": "ok",
                    "valkey_cache": "ok",
                }))
            }),
        );
        let client = spawn_mock_gateway(router).await;

        let health = client.health().await.expect("health should succeed");
        expect_that!(health["gateway"].as_str(), some(eq("ok")));
        expect_that!(health["clickhouse"].as_str(), some(eq("ok")));
    }

    #[gtest]
    #[tokio::test]
    async fn health_unhealthy_dependency_surfaces_503() {
        let router = Router::new().route(
            "/health",
            get(|| async {
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(json!({
                        "gateway": "ok",
                        "clickhouse": "error",
                        "postgres": "ok",
                        "valkey": "ok",
                        "valkey_cache": "ok",
                    })),
                )
            }),
        );
        let client = spawn_mock_gateway(router).await;

        let err = client
            .health()
            .await
            .expect_err("unhealthy gateway should error");
        match err {
            TensorZeroError::Http {
                status_code,
                text: Some(text),
                ..
            } => {
                expect_that!(status_code, eq(503));
                let body: Value =
                    serde_json::from_str(&text).expect("503 body should carry the health JSON");
                expect_that!(body["clickhouse"].as_str(), some(eq("error")));
            }
            other => panic!("expected a 503 HTTP error carrying the health body, got {other:?}"),
        }
    }

    // ------------------------------------------------------------------
    // Embedded mode
    // ------------------------------------------------------------------

    #[gtest]
    fn async_methods_reject_embedded_mode() {
        // `ClientBuilder` cannot construct an embedded client without config
        // and databases, so exercise the error helper directly.
        let err = http_only_error("submit_async_inference");
        expect_that!(
            err.to_string(),
            contains_substring("not supported in embedded gateway mode")
        );
    }

    // ------------------------------------------------------------------
    // Live-gateway e2e tests (Delta-AI fork)
    //
    // Gated on the `TZ_E2E_KEY` and `TZ_E2E_GATEWAY` environment variables;
    // every test skips when either is unset. The API key must only ever be
    // provided via the environment, never committed to the repository.
    // ------------------------------------------------------------------
    mod e2e {
        use super::*;

        const TEST_MODEL: &str = "deepseek-v4-flash";

        /// Builds an HTTP-mode client for the live gateway, or `None` (skip)
        /// when the e2e environment variables are not set.
        #[expect(clippy::print_stdout)]
        fn e2e_client() -> Option<Client> {
            let Ok(api_key) = std::env::var("TZ_E2E_KEY") else {
                println!("Skipping: TZ_E2E_KEY is not set");
                return None;
            };
            let Ok(gateway) = std::env::var("TZ_E2E_GATEWAY") else {
                println!("Skipping: TZ_E2E_GATEWAY is not set");
                return None;
            };
            let url = Url::parse(&gateway).expect("TZ_E2E_GATEWAY should be a valid URL");
            let client = ClientBuilder::new(ClientBuilderMode::HTTPGateway { url })
                .with_api_key(api_key)
                .build_http()
                .expect("e2e client should build");
            Some(client)
        }

        fn e2e_wait_options() -> AsyncTaskWaitOptions {
            // Keep the total budget under the nextest slow-timeout (30s of
            // silence kills the test).
            AsyncTaskWaitOptions {
                initial_interval: Duration::from_millis(500),
                max_interval: Duration::from_secs(2),
                timeout: Duration::from_secs(20),
            }
        }

        fn chat_body(model: &str) -> Value {
            json!({
                "model": model,
                "messages": [{"role": "user", "content": "Say hello in one word."}],
            })
        }

        #[gtest]
        #[tokio::test]
        async fn e2e_status_and_health() {
            let Some(client) = e2e_client() else {
                return;
            };
            let status = client.status().await.expect("status should succeed");
            expect_that!(status.status, eq("ok"));
            let health = client.health().await.expect("health should succeed");
            expect_that!(health["gateway"].as_str(), some(eq("ok")));
        }

        #[gtest]
        #[tokio::test]
        async fn e2e_submit_chat_and_wait_completed() {
            let Some(client) = e2e_client() else {
                return;
            };
            let launch = client
                .submit_async_inference(AsyncInferenceApiKind::Chat, chat_body(TEST_MODEL))
                .await
                .expect("submit should succeed");

            let status = client
                .wait_for_async_task(launch.task_id, e2e_wait_options())
                .await
                .expect("task should reach a terminal state");
            let AsyncTaskStatusResponse::Completed { task_id, response } = status else {
                panic!("expected the task to complete, got {status:?}");
            };
            expect_that!(task_id, eq(launch.task_id));
            expect_that!(
                response["object"].as_str(),
                some(eq("chat.completion")),
                "completed response should have the chat completions shape: {response}"
            );
            let content = response["choices"][0]["message"]["content"]
                .as_str()
                .expect("completed response should carry string content");
            expect_that!(content, not(eq("")));
        }

        #[gtest]
        #[tokio::test]
        async fn e2e_poll_observes_valid_statuses() {
            let Some(client) = e2e_client() else {
                return;
            };
            let launch = client
                .submit_async_inference(AsyncInferenceApiKind::Chat, chat_body(TEST_MODEL))
                .await
                .expect("submit should succeed");

            let mut seen = Vec::new();
            let terminal = loop {
                let status = client
                    .get_async_task(launch.task_id)
                    .await
                    .expect("status fetch should succeed");
                let name = match &status {
                    AsyncTaskStatusResponse::Queued { .. } => "queued",
                    AsyncTaskStatusResponse::Running { .. } => "running",
                    AsyncTaskStatusResponse::Completed { .. } => "completed",
                    AsyncTaskStatusResponse::Failed { .. } => "failed",
                    AsyncTaskStatusResponse::Cancelled { .. } => "cancelled",
                };
                seen.push(name.to_string());
                if is_terminal(&status) {
                    break status;
                }
                tokio::time::sleep(Duration::from_millis(300)).await;
                assert!(
                    seen.len() <= 60,
                    "task did not reach a terminal state in time; seen: {seen:?}"
                );
            };
            for name in &seen {
                expect_that!(
                    matches!(name.as_str(), "queued" | "running" | "completed"),
                    eq(true),
                    "only non-error statuses are expected for a good model; seen: {seen:?}"
                );
            }
            assert_that!(
                terminal,
                matches_pattern!(AsyncTaskStatusResponse::Completed { .. }),
                "final status should be completed; seen: {seen:?}"
            );
        }

        #[gtest]
        #[tokio::test]
        async fn e2e_stream_replays_chunks_and_ends_quietly() {
            let Some(client) = e2e_client() else {
                return;
            };
            let launch = client
                .submit_async_inference(AsyncInferenceApiKind::Chat, chat_body(TEST_MODEL))
                .await
                .expect("submit should succeed");

            // Attach immediately; the endpoint replays already-written events,
            // so this works whether the task is still running or just finished.
            // Retry the attach on two transient conditions observed against the
            // shared dev gateway: the gateway answers 500 when its Valkey read
            // of the event stream exceeds the command timeout, and a stream
            // that ends quietly with zero events means the connection was
            // dropped while the task was still queued (e.g. by an ingress idle
            // timeout).
            let mut events: Vec<_> = Vec::new();
            for attempt in 0..6 {
                if let Ok(stream) = client.stream_async_task(launch.task_id).await {
                    events = stream.collect().await;
                    if !events.is_empty() {
                        break;
                    }
                }
                tokio::time::sleep(Duration::from_secs((attempt + 1).min(4))).await;
            }

            assert_that!(
                events.len(),
                ge(1),
                "stream should yield at least one chunk event"
            );
            let mut content = String::new();
            for (index, event) in events.iter().enumerate() {
                let event = event
                    .as_ref()
                    .expect("stream should not produce error items for a good model");
                expect_that!(
                    event.data.as_str(),
                    not(eq("[DONE]")),
                    "the [DONE] sentinel should end the stream without being yielded (event {index})"
                );
                let chunk: Value =
                    serde_json::from_str(&event.data).expect("chunk data should be valid JSON");
                expect_that!(
                    chunk["object"].as_str(),
                    some(eq("chat.completion.chunk")),
                    "chunk should have the sync streaming shape: {chunk}"
                );
                if let Some(delta) = chunk["choices"][0]["delta"]["content"].as_str() {
                    content.push_str(delta);
                }
            }
            expect_that!(content.as_str(), not(eq("")));
        }

        #[gtest]
        #[tokio::test]
        async fn e2e_unknown_task_404_and_bad_model_fails() {
            let Some(client) = e2e_client() else {
                return;
            };

            let err = client
                .get_async_task(Uuid::now_v7())
                .await
                .expect_err("unknown task should error");
            match err {
                TensorZeroError::Http { status_code, .. } => expect_that!(status_code, eq(404)),
                other => panic!("expected a 404 HTTP error, got {other:?}"),
            }

            let launch = client
                .submit_async_inference(AsyncInferenceApiKind::Chat, chat_body("gpt-4o"))
                .await
                .expect("submit of an unknown model is still accepted (fails asynchronously)");
            let status = client
                .wait_for_async_task(launch.task_id, e2e_wait_options())
                .await
                .expect("task should reach a terminal state");
            let AsyncTaskStatusResponse::Failed { error, .. } = status else {
                panic!("expected the bad-model task to fail, got {status:?}");
            };
            let error = error.expect("a failed task should carry an error payload");
            expect_that!(
                error.to_string(),
                contains_substring("not found in model table"),
            );
        }
    }
}

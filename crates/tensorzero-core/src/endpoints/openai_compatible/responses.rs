// Modified by Delta-AI under Apache 2.0
//! OpenAI Responses API handler (`POST /v1/responses`).

use axum::Extension;
use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use serde_json::{Value, json};

use crate::endpoints::inference::{InferenceOutput, InferenceResponseChunk};
use crate::endpoints::openai_compatible::types::responses::{
    OpenAICompatibleResponsesParams, OpenAICompatibleResponsesResponse, responses_output_items,
};
use crate::error::{Error, ErrorDetails};
use crate::inference::types::{ContentBlockChunk, Usage, current_timestamp};
use crate::utils::gateway::AppState;
use tensorzero_auth::middleware::RequestApiKeyExtension;

use super::infer::{error_response, infer_openai_compatible};
use super::synapse::SynapseRequestContext;
use super::{OpenAICompatibleError, OpenAIStructuredJson};

pub async fn responses_handler(
    State(state): AppState,
    api_key_ext: Option<Extension<RequestApiKeyExtension>>,
    headers: HeaderMap,
    OpenAIStructuredJson(params): OpenAIStructuredJson<OpenAICompatibleResponsesParams>,
) -> Result<Response, OpenAICompatibleError> {
    let synapse = SynapseRequestContext::from_headers(&headers);
    let chat_params = match params.into_chat_params() {
        Ok(chat_params) => chat_params,
        Err(error) => return Ok(error_response(error, false, &synapse)),
    };

    let inferred = match Box::pin(infer_openai_compatible(
        &state,
        api_key_ext,
        &headers,
        chat_params,
    ))
    .await
    {
        Ok(inferred) => inferred,
        Err(response) => return Ok(response),
    };

    let mut response = match inferred.output {
        InferenceOutput::NonStreaming(response) => {
            let body =
                OpenAICompatibleResponsesResponse::from((response, inferred.response_model_prefix));
            Json(body).into_response()
        }
        InferenceOutput::Streaming(stream) => {
            let responses_stream =
                prepare_serialized_openai_responses_events(stream, inferred.response_model_prefix);
            Sse::new(responses_stream)
                .keep_alive(axum::response::sse::KeepAlive::new())
                .into_response()
        }
    };
    inferred.synapse.apply_to_response(&mut response);
    Ok(response)
}

fn prepare_serialized_openai_responses_events(
    mut stream: crate::endpoints::inference::InferenceStream,
    response_model_prefix: String,
) -> impl futures::Stream<Item = Result<Event, Error>> {
    async_stream::stream! {
        let mut accumulated = String::new();
        let mut created_emitted = false;
        let mut response_id = String::new();
        let mut message_id = String::new();
        let mut model = String::new();
        let mut last_usage: Option<Usage> = None;

        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(e) => {
                    let error_event = e.build_streaming_error_event(true, false);
                    yield Event::default().json_data(&error_event).map_err(|ser_err| {
                        Error::new(ErrorDetails::Inference {
                            message: format!("Failed to convert error to Event: {ser_err}"),
                        })
                    });
                    continue;
                }
            };

            let (inference_id, variant_name, text_delta, usage) = match chunk {
                InferenceResponseChunk::Chat(c) => {
                    let text = c.content.into_iter().find_map(|block| match block {
                        ContentBlockChunk::Text(text) => Some(text.text),
                        _ => None,
                    });
                    (c.inference_id, c.variant_name, text, c.usage)
                }
                InferenceResponseChunk::Json(c) => {
                    let text = if c.raw.is_empty() {
                        None
                    } else {
                        Some(c.raw)
                    };
                    (c.inference_id, c.variant_name, text, c.usage)
                }
            };

            if !created_emitted {
                response_id = format!("resp_{inference_id}");
                message_id = format!("msg_{inference_id}");
                model = format!("{response_model_prefix}{variant_name}");
                let created_payload = json!({
                    "type": "response.created",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "created_at": current_timestamp(),
                        "status": "in_progress",
                        "model": model,
                        "output": [],
                    }
                });
                yield sse_named_event("response.created", &created_payload);
                created_emitted = true;
            }

            if let Some(delta) = text_delta.filter(|value| !value.is_empty()) {
                accumulated.push_str(&delta);
                let delta_payload = json!({
                    "type": "response.output_text.delta",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": delta,
                });
                yield sse_named_event("response.output_text.delta", &delta_payload);
            }

            if usage.is_some() {
                last_usage = usage;
            }
        }

        if created_emitted {
            let usage = last_usage.unwrap_or_default();
            let completed = json!({
                "type": "response.completed",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": current_timestamp(),
                    "status": "completed",
                    "model": model,
                    "output": responses_output_items(
                        &message_id,
                        Some(accumulated.as_str()).filter(|value| !value.is_empty()),
                        &[],
                    ),
                    "usage": {
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "total_tokens": usage.total_tokens(),
                    },
                }
            });
            yield sse_named_event("response.completed", &completed);
        }
    }
}

fn sse_named_event(event_name: &'static str, payload: &Value) -> Result<Event, Error> {
    Event::default()
        .event(event_name)
        .json_data(payload)
        .map_err(|e| {
            Error::new(ErrorDetails::Inference {
                message: format!("Failed to convert chunk to Event: {e}"),
            })
        })
}

// Modified by Delta-AI under Apache 2.0
//! OpenAI Completions API handler (`POST /v1/completions`).

use axum::Extension;
use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use serde::Serialize;

use crate::endpoints::inference::{InferenceOutput, InferenceResponseChunk};
use crate::endpoints::openai_compatible::types::chat_completions::OpenAICompatibleFinishReason;
use crate::endpoints::openai_compatible::types::streaming::process_chat_content_chunk;
use crate::error::{Error, ErrorDetails};
use crate::inference::types::current_timestamp;
use crate::utils::gateway::AppState;
use tensorzero_auth::middleware::RequestApiKeyExtension;

use super::infer::{error_response, infer_openai_compatible};
use super::synapse::SynapseRequestContext;
use super::types::completions::{
    OpenAICompatibleCompletionParams, OpenAICompatibleCompletionResponse,
};
use super::types::streaming::StreamingContentState;
use super::types::usage::OpenAICompatibleUsage;
use super::{OpenAICompatibleError, OpenAIStructuredJson};

pub async fn completions_handler(
    State(state): AppState,
    api_key_ext: Option<Extension<RequestApiKeyExtension>>,
    headers: HeaderMap,
    OpenAIStructuredJson(params): OpenAIStructuredJson<OpenAICompatibleCompletionParams>,
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
            let body = OpenAICompatibleCompletionResponse::from((
                response,
                inferred.response_model_prefix,
            ));
            Json(body).into_response()
        }
        InferenceOutput::Streaming(stream) => {
            let completion_stream = prepare_serialized_openai_completion_events(
                stream,
                inferred.response_model_prefix,
                inferred.include_usage,
            );
            Sse::new(completion_stream)
                .keep_alive(axum::response::sse::KeepAlive::new())
                .into_response()
        }
    };
    inferred.synapse.apply_to_response(&mut response);
    Ok(response)
}

#[derive(Clone, Debug, Serialize)]
struct OpenAICompatibleCompletionChunk {
    id: String,
    object: String,
    created: u32,
    model: String,
    choices: Vec<OpenAICompatibleCompletionChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<OpenAICompatibleUsage>,
}

#[derive(Clone, Debug, Serialize)]
struct OpenAICompatibleCompletionChunkChoice {
    text: String,
    index: u32,
    logprobs: Option<()>,
    finish_reason: Option<OpenAICompatibleFinishReason>,
}

fn prepare_serialized_openai_completion_events(
    mut stream: crate::endpoints::inference::InferenceStream,
    response_model_prefix: String,
    include_usage: bool,
) -> impl futures::Stream<Item = Result<Event, Error>> {
    async_stream::stream! {
        let mut state = StreamingContentState::default();
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
            let completion_chunk = match chunk {
                InferenceResponseChunk::Chat(c) => {
                    let (content, _tool_calls, _extra) =
                        process_chat_content_chunk(c.content, &mut state);
                    OpenAICompatibleCompletionChunk {
                        id: c.inference_id.to_string(),
                        object: "text_completion".to_string(),
                        created: current_timestamp() as u32,
                        model: format!("{response_model_prefix}{}", c.variant_name),
                        choices: vec![OpenAICompatibleCompletionChunkChoice {
                            text: content.unwrap_or_default(),
                            index: 0,
                            logprobs: None,
                            finish_reason: c.finish_reason.map(OpenAICompatibleFinishReason::from),
                        }],
                        usage: if include_usage {
                            c.usage.map(OpenAICompatibleUsage::from)
                        } else {
                            None
                        },
                    }
                }
                InferenceResponseChunk::Json(c) => {
                    OpenAICompatibleCompletionChunk {
                        id: c.inference_id.to_string(),
                        object: "text_completion".to_string(),
                        created: current_timestamp() as u32,
                        model: format!("{response_model_prefix}{}", c.variant_name),
                        choices: vec![OpenAICompatibleCompletionChunkChoice {
                            text: c.raw,
                            index: 0,
                            logprobs: None,
                            finish_reason: c.finish_reason.map(OpenAICompatibleFinishReason::from),
                        }],
                        usage: if include_usage {
                            c.usage.map(OpenAICompatibleUsage::from)
                        } else {
                            None
                        },
                    }
                }
            };
            yield Event::default().json_data(&completion_chunk).map_err(|e| {
                Error::new(ErrorDetails::Inference {
                    message: format!("Failed to convert chunk to Event: {e}"),
                })
            });
        }
        yield Ok(Event::default().data("[DONE]"));
    }
}

// Modified by Delta-AI under Apache 2.0
//! Chat completions endpoint handler for OpenAI-compatible API.
//!
//! This module implements the HTTP handler for the `/openai/v1/chat/completions`
//! and `/v1/chat/completions` endpoints.

use axum::Extension;
use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::Sse;
use axum::response::{IntoResponse, Response};
use futures::StreamExt;

use crate::endpoints::inference::{InferenceOutput, InferenceResponse};
use crate::utils::gateway::{AppState, AppStateData};
use tensorzero_auth::middleware::RequestApiKeyExtension;

use super::anthropic_messages::{anthropic_from_inference, prepare_anthropic_sse};
use super::infer::infer_openai_compatible;
use super::types::chat_completions::{OpenAICompatibleParams, OpenAICompatibleResponse};
use super::types::streaming::{SerializedSseEvent, prepare_serialized_openai_compatible_events};
use super::{OpenAICompatibleError, OpenAIStructuredJson};

/// A handler for the OpenAI-compatible inference endpoint
pub async fn chat_completions_handler(
    State(state): AppState,
    api_key_ext: Option<Extension<RequestApiKeyExtension>>,
    headers: HeaderMap,
    OpenAIStructuredJson(openai_compatible_params): OpenAIStructuredJson<OpenAICompatibleParams>,
) -> Result<Response, OpenAICompatibleError> {
    Box::pin(handle_chat_completions(
        &state,
        api_key_ext,
        &headers,
        openai_compatible_params,
    ))
    .await
}

pub(super) async fn handle_chat_completions(
    state: &AppStateData,
    api_key_ext: Option<Extension<RequestApiKeyExtension>>,
    headers: &HeaderMap,
    openai_compatible_params: OpenAICompatibleParams,
) -> Result<Response, OpenAICompatibleError> {
    let inferred = match Box::pin(infer_openai_compatible(
        state,
        api_key_ext,
        headers,
        openai_compatible_params,
    ))
    .await
    {
        Ok(inferred) => inferred,
        Err(response) => return Ok(response),
    };

    let mut response = match inferred.output {
        InferenceOutput::NonStreaming(response) => {
            if inferred.synapse.response_style_anthropic {
                let variant_name = match &response {
                    InferenceResponse::Chat(chat) => chat.variant_name.clone(),
                    InferenceResponse::Json(json) => json.variant_name.clone(),
                };
                let model = format!("{}{variant_name}", inferred.response_model_prefix);
                Json(anthropic_from_inference(response, &model)).into_response()
            } else {
                let openai_compatible_response = OpenAICompatibleResponse::from((
                    response,
                    inferred.response_model_prefix,
                    inferred.include_original_response,
                    inferred.include_raw_response,
                ));
                Json(openai_compatible_response).into_response()
            }
        }
        InferenceOutput::Streaming(stream) => {
            if inferred.synapse.response_style_anthropic {
                let events = prepare_anthropic_sse(
                    stream,
                    inferred.response_model_prefix,
                    inferred.synapse.stream_aggregate.clone(),
                )
                .map(|frame| frame.map(SerializedSseEvent::into_event));
                Sse::new(events)
                    .keep_alive(axum::response::sse::KeepAlive::new())
                    .into_response()
            } else {
                let openai_compatible_stream = prepare_serialized_openai_compatible_events(
                    stream,
                    inferred.response_model_prefix,
                    inferred.include_usage,
                    inferred.include_raw_usage,
                    inferred.include_original_response,
                    inferred.include_raw_response,
                    inferred.synapse.stream_aggregate.clone(),
                )
                .map(|frame| frame.map(SerializedSseEvent::into_event));
                Sse::new(openai_compatible_stream)
                    .keep_alive(axum::response::sse::KeepAlive::new())
                    .into_response()
            }
        }
    };
    inferred.synapse.apply_to_response(&mut response);
    Ok(response)
}

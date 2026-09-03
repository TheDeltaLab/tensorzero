// Modified by Delta-AI under Apache 2.0
//! Inbound Anthropic Messages API (`POST /v1/messages`, `POST /anthropic/v1/messages`).
//!
//! Claude Code hits these paths. TensorZero converts the wire format to internal
//! inference and emits Anthropic-shaped JSON / SSE so callers do not change.

use std::time::Instant;

use axum::Extension;
use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::Sse;
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use mime::MediaType;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tensorzero_auth::middleware::RequestApiKeyExtension;
use tensorzero_inference_types::tool::{DynamicToolParams, FunctionTool, Tool};
use tokio::time::sleep;

use crate::cache::CacheParamsOptions;
use crate::endpoints::inference::{
    ChatCompletionInferenceParams, ChatInferenceResponse, InferenceOutput, InferenceParams,
    InferenceResponse, InferenceResponseChunk, InferenceStream, JsonInferenceResponse, Params,
    inference,
};
use crate::error::{Error, ErrorDetails};
use crate::inference::types::{
    Base64File, ContentBlockChatOutput, File, FinishReason, Input, InputMessage,
    InputMessageContent, Role, System, Text, Thought, UrlFile,
};
use crate::routing::RoutingSession;
use crate::tool::{ToolCall, ToolCallWrapper, ToolChoice, ToolResult};
use crate::utils::gateway::{AppState, AppStateData};

use super::infer::error_response;
use super::stream_aggregator::{StreamAggregateRule, StreamAggregator};
use super::synapse::{
    SynapseRequestContext, apply_compat_to_params, resolve_cache_options,
    resolve_openai_compatible_model, run_with_request_timeout,
};
use super::types::streaming::{SerializedSseEvent, value_to_sse_frame};
use super::{OpenAICompatibleError, OpenAIStructuredJson};

#[derive(Debug, Deserialize)]
pub struct AnthropicMessagesParams {
    pub model: String,
    #[serde(default)]
    pub messages: Vec<AnthropicIncomingMessage>,
    #[serde(default)]
    pub system: Option<AnthropicSystem>,
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub tools: Option<Vec<AnthropicToolDef>>,
    #[serde(default)]
    pub tool_choice: Option<AnthropicToolChoice>,
    #[serde(default)]
    pub thinking: Option<AnthropicThinking>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum AnthropicSystem {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

#[derive(Debug, Deserialize)]
pub struct AnthropicIncomingMessage {
    pub role: String,
    pub content: AnthropicContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: AnthropicImageSource },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: Option<AnthropicToolResultContent>,
    },
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        #[serde(default)]
        signature: Option<String>,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicImageSource {
    #[serde(rename = "url")]
    Url { url: String },
    #[serde(rename = "base64")]
    Base64 { media_type: String, data: String },
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum AnthropicToolResultContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

#[derive(Debug, Deserialize)]
pub struct AnthropicToolDef {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub input_schema: Value,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicToolChoice {
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "any")]
    Any,
    #[serde(rename = "none")]
    None,
    #[serde(rename = "tool")]
    Tool { name: String },
}

#[derive(Debug, Deserialize)]
pub struct AnthropicThinking {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub budget_tokens: Option<i32>,
}

#[derive(Debug, Serialize)]
pub struct AnthropicMessageResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub role: &'static str,
    pub model: String,
    pub content: Vec<Value>,
    pub stop_reason: String,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

pub async fn messages_handler(
    State(state): AppState,
    api_key_ext: Option<Extension<RequestApiKeyExtension>>,
    headers: HeaderMap,
    OpenAIStructuredJson(params): OpenAIStructuredJson<AnthropicMessagesParams>,
) -> Result<Response, OpenAICompatibleError> {
    Box::pin(handle_anthropic_messages(
        &state,
        api_key_ext,
        &headers,
        params,
    ))
    .await
}

pub(super) async fn handle_anthropic_messages(
    state: &AppStateData,
    api_key_ext: Option<Extension<RequestApiKeyExtension>>,
    headers: &HeaderMap,
    params: AnthropicMessagesParams,
) -> Result<Response, OpenAICompatibleError> {
    let validated = match validate_anthropic_request(headers, params) {
        Ok(validated) => validated,
        Err(error) => {
            return Ok(error_response(
                error,
                false,
                &SynapseRequestContext::from_headers(headers),
            ));
        }
    };
    let response_model = validated.response_model.clone();
    let stream_aggregate = validated.synapse.stream_aggregate.clone();

    let (output, synapse) = match Box::pin(execute_anthropic(state, api_key_ext, validated)).await {
        Ok(ok) => ok,
        Err(rejection) => {
            return Ok(error_response(rejection.error, false, &rejection.synapse));
        }
    };

    let mut response = match output {
        InferenceOutput::NonStreaming(response) => {
            Json(anthropic_from_inference(response, &response_model)).into_response()
        }
        InferenceOutput::Streaming(stream) => {
            let events = prepare_anthropic_sse(stream, response_model, stream_aggregate)
                .map(|frame| frame.map(SerializedSseEvent::into_event));
            Sse::new(events)
                .keep_alive(axum::response::sse::KeepAlive::new())
                .into_response()
        }
    };
    synapse.apply_to_response(&mut response);
    Ok(response)
}

/// An Anthropic Messages request that passed validation and is ready to
/// execute via [`execute_anthropic`].
pub(super) struct AnthropicValidatedRequest {
    pub tz_params: Params,
    pub synapse: SynapseRequestContext,
    pub response_model: String,
}

/// Anthropic execution failure (the inference itself errored).
pub(super) struct AnthropicExecutionError {
    pub error: Error,
    pub synapse: SynapseRequestContext,
}

/// Validate an Anthropic Messages body and apply Synapse headers, without
/// running inference. Shared between the synchronous handler and the async
/// inference submit endpoint, which validates at submit time so obvious
/// errors surface synchronously.
pub(super) fn validate_anthropic_request(
    headers: &HeaderMap,
    params: AnthropicMessagesParams,
) -> Result<AnthropicValidatedRequest, Error> {
    let mut synapse = SynapseRequestContext::try_from_headers(headers)?;
    let mut tz_params = params_from_anthropic(params, &synapse)?;
    tz_params.extra_internal_tags = synapse.observability_tags(headers);
    apply_compat_to_params(headers, &mut tz_params)?;
    synapse = synapse.with_served_by_from_params(&tz_params);
    let response_model = tz_params
        .model_name
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    Ok(AnthropicValidatedRequest {
        tz_params,
        synapse,
        response_model,
    })
}

/// Execute a validated Anthropic Messages request against [`inference`],
/// applying the Synapse per-request timeout and recording routing outcomes
/// on the Synapse context.
pub(super) async fn execute_anthropic(
    state: &AppStateData,
    api_key_ext: Option<Extension<RequestApiKeyExtension>>,
    validated: AnthropicValidatedRequest,
) -> Result<(InferenceOutput, SynapseRequestContext), AnthropicExecutionError> {
    let AnthropicValidatedRequest {
        tz_params,
        mut synapse,
        ..
    } = validated;
    let session = RoutingSession::new(synapse.fallback_disabled);
    let inference_result = Box::pin(run_with_request_timeout(
        synapse.request_timeout,
        RoutingSession::scope(session.clone(), async {
            Box::pin(inference(
                state.config.clone(),
                &state.http_client,
                state.clickhouse_connection_info.clone(),
                state.postgres_connection_info.clone(),
                state.cache_manager.clone(),
                state.deferred_tasks.clone(),
                state.rate_limiting_manager.clone(),
                state.primary_datastore,
                tz_params,
                api_key_ext,
            ))
            .await
        }),
    ))
    .await;

    if let Some(outcome) = session.take_outcome() {
        synapse.served_by = Some(outcome.served_by);
        synapse.fallback_count = outcome.fallback_count;
    }

    match inference_result {
        Ok(data) => Ok((data.output, synapse)),
        Err(error) => Err(AnthropicExecutionError { error, synapse }),
    }
}

fn params_from_anthropic(
    params: AnthropicMessagesParams,
    synapse: &SynapseRequestContext,
) -> Result<Params, Error> {
    let model = resolve_openai_compatible_model(&params.model, synapse.provider.as_deref())?;
    let input = anthropic_to_input(params.system, params.messages)?;
    let additional_tools = params.tools.map(|tools| {
        tools
            .into_iter()
            .map(|tool| {
                Tool::Function(FunctionTool {
                    name: tool.name,
                    description: tool.description.unwrap_or_default(),
                    parameters: tool.input_schema,
                    strict: false,
                })
            })
            .collect()
    });
    let tool_choice = params.tool_choice.map(|choice| match choice {
        AnthropicToolChoice::Auto => ToolChoice::Auto,
        AnthropicToolChoice::Any => ToolChoice::Required,
        AnthropicToolChoice::None => ToolChoice::None,
        AnthropicToolChoice::Tool { name } => ToolChoice::Specific(name),
    });
    let thinking_budget_tokens = params.thinking.and_then(|thinking| {
        if thinking.kind == "enabled" {
            thinking.budget_tokens
        } else {
            None
        }
    });
    Ok(Params {
        model_name: Some(model),
        input,
        stream: params.stream,
        params: InferenceParams {
            chat_completion: ChatCompletionInferenceParams {
                max_tokens: params.max_tokens,
                temperature: params.temperature,
                top_p: params.top_p,
                stop_sequences: params.stop_sequences,
                thinking_budget_tokens,
                ..Default::default()
            },
        },
        dynamic_tool_params: DynamicToolParams {
            additional_tools,
            tool_choice,
            ..Default::default()
        },
        cache_options: resolve_cache_options(None::<CacheParamsOptions>, synapse.cache_disabled),
        ..Default::default()
    })
}

fn flatten_system_text(system: Option<AnthropicSystem>) -> Option<String> {
    match system {
        Some(AnthropicSystem::Text(text)) => {
            if text.is_empty() {
                None
            } else {
                Some(text)
            }
        }
        Some(AnthropicSystem::Blocks(blocks)) => {
            let text = system_text_from_blocks(&blocks);
            if text.is_empty() { None } else { Some(text) }
        }
        None => None,
    }
}

fn system_text_from_blocks(blocks: &[AnthropicContentBlock]) -> String {
    let mut text = String::new();
    for block in blocks {
        if let AnthropicContentBlock::Text { text: piece } = block {
            text.push_str(piece);
        }
    }
    text
}

fn system_text_from_content(content: AnthropicContent) -> String {
    match content {
        AnthropicContent::Text(text) => text,
        AnthropicContent::Blocks(blocks) => system_text_from_blocks(&blocks),
    }
}

fn anthropic_to_input(
    system: Option<AnthropicSystem>,
    messages: Vec<AnthropicIncomingMessage>,
) -> Result<Input, Error> {
    // Official Anthropic puts the prompt on the top-level `system` field.
    // Claude Code (and some proxies) also send `{ "role": "system" }` inside
    // `messages`. Lift those into `Input.system` the same way the OpenAI
    // chat-completions path does, instead of 400ing.
    let mut system_parts = Vec::new();
    if let Some(text) = flatten_system_text(system) {
        system_parts.push(text);
    }
    let mut out = Vec::new();
    for message in messages {
        if message.role == "system" {
            let text = system_text_from_content(message.content);
            if !text.is_empty() {
                system_parts.push(text);
            }
            continue;
        }
        let role = match message.role.as_str() {
            "user" => Role::User,
            "assistant" => Role::Assistant,
            other => {
                return Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                    message: format!("Unsupported Anthropic role `{other}`"),
                }));
            }
        };
        let content = match message.content {
            AnthropicContent::Text(text) => vec![InputMessageContent::Text(Text { text })],
            AnthropicContent::Blocks(blocks) => {
                let mut content = Vec::new();
                for block in blocks {
                    if let Some(item) = convert_block(block)? {
                        content.push(item);
                    }
                }
                content
            }
        };
        out.push(InputMessage { role, content });
    }
    let system = if system_parts.is_empty() {
        None
    } else {
        Some(System::Text(system_parts.join("\n")))
    };
    Ok(Input {
        system,
        messages: out,
    })
}

fn convert_block(block: AnthropicContentBlock) -> Result<Option<InputMessageContent>, Error> {
    Ok(Some(match block {
        AnthropicContentBlock::Text { text } => InputMessageContent::Text(Text { text }),
        AnthropicContentBlock::Image { source } => match source {
            AnthropicImageSource::Url { url } => {
                let url = url::Url::parse(&url).map_err(|e| {
                    Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                        message: format!("Invalid image URL: {e}"),
                    })
                })?;
                InputMessageContent::File(File::Url(UrlFile {
                    url,
                    mime_type: None,
                    detail: None,
                    filename: None,
                }))
            }
            AnthropicImageSource::Base64 { media_type, data } => {
                let mime_type: MediaType = media_type.parse().map_err(|_| {
                    Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                        message: format!("Unknown image MIME type `{media_type}`"),
                    })
                })?;
                InputMessageContent::File(File::Base64(Base64File::new(
                    None,
                    Some(mime_type),
                    data,
                    None,
                    None,
                )?))
            }
        },
        AnthropicContentBlock::ToolUse { id, name, input } => {
            InputMessageContent::ToolCall(ToolCallWrapper::ToolCall(ToolCall {
                id,
                name,
                arguments: input.to_string(),
            }))
        }
        AnthropicContentBlock::ToolResult {
            tool_use_id,
            content,
        } => {
            let result = match content {
                Some(AnthropicToolResultContent::Text(text)) => text,
                Some(AnthropicToolResultContent::Blocks(blocks)) => blocks
                    .into_iter()
                    .filter_map(|block| match block {
                        AnthropicContentBlock::Text { text } => Some(text),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join(""),
                None => String::new(),
            };
            InputMessageContent::ToolResult(ToolResult {
                id: tool_use_id,
                name: String::new(),
                result,
            })
        }
        AnthropicContentBlock::Thinking {
            thinking,
            signature,
        } => InputMessageContent::Thought(Thought {
            text: Some(thinking),
            signature,
            summary: None,
            provider_type: Some("anthropic".to_string()),
            extra_data: None,
        }),
        AnthropicContentBlock::Other => return Ok(None),
    }))
}

pub fn anthropic_from_inference(
    response: InferenceResponse,
    model: &str,
) -> AnthropicMessageResponse {
    match response {
        InferenceResponse::Chat(chat) => anthropic_from_chat(chat, model),
        InferenceResponse::Json(json_resp) => anthropic_from_json(json_resp, model),
    }
}

fn anthropic_from_json(json_resp: JsonInferenceResponse, model: &str) -> AnthropicMessageResponse {
    AnthropicMessageResponse {
        id: format!("msg_{}", json_resp.inference_id),
        kind: "message",
        role: "assistant",
        model: model.to_string(),
        content: json_resp
            .output
            .raw
            .map(|text| vec![json!({"type": "text", "text": text})])
            .unwrap_or_default(),
        stop_reason: stop_reason(json_resp.finish_reason),
        usage: AnthropicUsage {
            input_tokens: json_resp.usage.input_tokens.unwrap_or(0),
            output_tokens: json_resp.usage.output_tokens.unwrap_or(0),
        },
    }
}

fn anthropic_from_chat(chat: ChatInferenceResponse, model: &str) -> AnthropicMessageResponse {
    let mut content = Vec::new();
    for block in chat.content {
        match block {
            ContentBlockChatOutput::Text(Text { text }) => {
                content.push(json!({"type": "text", "text": text}));
            }
            ContentBlockChatOutput::Thought(thought) => {
                let mut obj =
                    json!({"type": "thinking", "thinking": thought.text.unwrap_or_default()});
                if let Some(signature) = thought.signature {
                    obj["signature"] = json!(signature);
                }
                content.push(obj);
            }
            ContentBlockChatOutput::ToolCall(tool) => {
                let input = tool.arguments.clone().unwrap_or_else(|| {
                    serde_json::from_str(&tool.raw_arguments).unwrap_or(json!({}))
                });
                content.push(json!({
                    "type": "tool_use",
                    "id": tool.id,
                    "name": tool.name.unwrap_or(tool.raw_name),
                    "input": input,
                }));
            }
            ContentBlockChatOutput::Unknown(_) => {}
        }
    }
    AnthropicMessageResponse {
        id: format!("msg_{}", chat.inference_id),
        kind: "message",
        role: "assistant",
        model: model.to_string(),
        content,
        stop_reason: stop_reason(chat.finish_reason),
        usage: AnthropicUsage {
            input_tokens: chat.usage.input_tokens.unwrap_or(0),
            output_tokens: chat.usage.output_tokens.unwrap_or(0),
        },
    }
}

fn stop_reason(reason: Option<FinishReason>) -> String {
    match reason {
        Some(FinishReason::Length) => "max_tokens",
        Some(FinishReason::ToolCall) => "tool_use",
        Some(FinishReason::StopSequence) => "stop_sequence",
        _ => "end_turn",
    }
    .to_string()
}

pub(super) fn prepare_anthropic_sse(
    mut stream: InferenceStream,
    model: String,
    aggregate: Option<Vec<StreamAggregateRule>>,
) -> impl futures::Stream<Item = Result<SerializedSseEvent, Error>> {
    async_stream::stream! {
        let mut started = false;
        let mut open_index: Option<usize> = None;
        let mut open_kind: Option<&'static str> = None;
        let mut next_index: usize = 0;
        let mut aggregator = aggregate.map(StreamAggregator::new);

        loop {
            let wait = aggregator
                .as_ref()
                .and_then(StreamAggregator::next_deadline)
                .map(|deadline| deadline.saturating_duration_since(Instant::now()));
            if wait.is_some_and(|duration| duration.is_zero()) {
                if let Some(agg) = aggregator.as_mut()
                    && let Some((event, value)) = agg.flush_if_due(Instant::now())
                {
                    yield Ok(value_to_sse_frame(event, value));
                }
                continue;
            }

            let chunk = if let Some(wait) = wait {
                tokio::select! {
                    chunk = stream.next() => chunk,
                    () = sleep(wait) => {
                        if let Some(agg) = aggregator.as_mut()
                            && let Some((event, value)) = agg.flush_if_due(Instant::now())
                        {
                            yield Ok(value_to_sse_frame(event, value));
                        }
                        continue;
                    }
                }
            } else {
                stream.next().await
            };

            let Some(chunk) = chunk else {
                break;
            };
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(e) => {
                    yield Err(e);
                    continue;
                }
            };
            let InferenceResponseChunk::Chat(chat) = chunk else {
                continue;
            };
            if !started {
                started = true;
                let start = json!({
                    "type": "message_start",
                    "message": {
                        "id": format!("msg_{}", chat.inference_id),
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [],
                        "usage": {
                            "input_tokens": chat.usage.as_ref().and_then(|u| u.input_tokens).unwrap_or(0),
                            "output_tokens": 0
                        }
                    }
                });
                yield Ok(SerializedSseEvent::new(
                    Some("message_start".to_string()),
                    start.to_string(),
                ));
            }
            for block in chat.content {
                match block {
                    crate::inference::types::ContentBlockChunk::Text(text) => {
                        if open_kind != Some("text") {
                            if let Some(index) = open_index {
                                yield Ok(emit_stop(index));
                            }
                            yield Ok(emit_start(next_index, json!({"type": "text", "text": ""})));
                            open_index = Some(next_index);
                            open_kind = Some("text");
                            next_index += 1;
                        }
                        let index = open_index.unwrap_or(0);
                        let payload = json!({
                            "type": "content_block_delta",
                            "index": index,
                            "delta": { "type": "text_delta", "text": text.text }
                        });
                        for event in emit_maybe_aggregate(&mut aggregator, Some("content_block_delta"), payload) {
                            yield event;
                        }
                    }
                    crate::inference::types::ContentBlockChunk::Thought(thought) => {
                        if open_kind != Some("thinking") {
                            if let Some(index) = open_index {
                                yield Ok(emit_stop(index));
                            }
                            yield Ok(emit_start(
                                next_index,
                                json!({"type": "thinking", "thinking": ""}),
                            ));
                            open_index = Some(next_index);
                            open_kind = Some("thinking");
                            next_index += 1;
                        }
                        if let Some(piece) = thought.text {
                            let index = open_index.unwrap_or(0);
                            let payload = json!({
                                "type": "content_block_delta",
                                "index": index,
                                "delta": { "type": "thinking_delta", "thinking": piece }
                            });
                            for event in emit_maybe_aggregate(&mut aggregator, Some("content_block_delta"), payload) {
                                yield event;
                            }
                        }
                    }
                    crate::inference::types::ContentBlockChunk::ToolCall(tool) => {
                        if open_kind != Some("tool_use") {
                            if let Some(index) = open_index {
                                yield Ok(emit_stop(index));
                            }
                            let name = tool.raw_name.unwrap_or_default();
                            let tool_start = json!({
                                "type": "tool_use",
                                "id": tool.id,
                                "name": name,
                                "input": {}
                            });
                            yield Ok(emit_start(next_index, tool_start));
                            open_index = Some(next_index);
                            open_kind = Some("tool_use");
                            next_index += 1;
                        }
                        if !tool.raw_arguments.is_empty() {
                            let index = open_index.unwrap_or(0);
                            let payload = json!({
                                "type": "content_block_delta",
                                "index": index,
                                "delta": { "type": "input_json_delta", "partial_json": tool.raw_arguments }
                            });
                            yield Ok(SerializedSseEvent::new(
                                Some("content_block_delta".to_string()),
                                payload.to_string(),
                            ));
                        }
                    }
                    crate::inference::types::ContentBlockChunk::Unknown(_) => {}
                }
            }
            if let Some(finish) = chat.finish_reason {
                if let Some(index) = open_index.take() {
                    yield Ok(emit_stop(index));
                }
                open_kind = None;
                let usage = chat.usage.unwrap_or_default();
                let delta = json!({
                    "type": "message_delta",
                    "delta": { "stop_reason": stop_reason(Some(finish)) },
                    "usage": { "output_tokens": usage.output_tokens.unwrap_or(0) }
                });
                yield Ok(SerializedSseEvent::new(
                    Some("message_delta".to_string()),
                    delta.to_string(),
                ));
                yield Ok(SerializedSseEvent::new(
                    Some("message_stop".to_string()),
                    json!({"type":"message_stop"}).to_string(),
                ));
            }
        }
        if let Some(agg) = aggregator.as_mut() {
            for (event, value) in agg.finish() {
                yield Ok(value_to_sse_frame(event, value));
            }
        }
        if let Some(index) = open_index.take() {
            yield Ok(emit_stop(index));
            yield Ok(SerializedSseEvent::new(
                Some("message_stop".to_string()),
                json!({"type":"message_stop"}).to_string(),
            ));
        }
    }
}

fn emit_start(index: usize, block: Value) -> SerializedSseEvent {
    let payload = json!({"type": "content_block_start", "index": index, "content_block": block});
    SerializedSseEvent::new(Some("content_block_start".to_string()), payload.to_string())
}

fn emit_stop(index: usize) -> SerializedSseEvent {
    let payload = json!({"type": "content_block_stop", "index": index});
    SerializedSseEvent::new(Some("content_block_stop".to_string()), payload.to_string())
}

fn emit_maybe_aggregate(
    aggregator: &mut Option<StreamAggregator>,
    event_name: Option<&str>,
    payload: Value,
) -> Vec<Result<SerializedSseEvent, Error>> {
    let Some(agg) = aggregator.as_mut() else {
        return vec![Ok(value_to_sse_frame(
            event_name.map(str::to_string),
            payload,
        ))];
    };
    let data = payload.to_string();
    agg.push(event_name, &data, Instant::now())
        .into_iter()
        .map(|(event, value)| Ok(value_to_sse_frame(event, value)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use googletest::prelude::*;

    #[test]
    fn converts_text_and_tools() {
        let params = AnthropicMessagesParams {
            model: "claude-sonnet-4-5".to_string(),
            messages: vec![AnthropicIncomingMessage {
                role: "user".to_string(),
                content: AnthropicContent::Text("hi".to_string()),
            }],
            system: Some(AnthropicSystem::Text("sys".to_string())),
            max_tokens: Some(32),
            stream: Some(false),
            temperature: None,
            top_p: None,
            stop_sequences: None,
            tools: Some(vec![AnthropicToolDef {
                name: "lookup".to_string(),
                description: Some("look up".to_string()),
                input_schema: json!({"type": "object"}),
            }]),
            tool_choice: Some(AnthropicToolChoice::Auto),
            thinking: Some(AnthropicThinking {
                kind: "enabled".to_string(),
                budget_tokens: Some(1024),
            }),
        };
        let synapse = SynapseRequestContext::from_headers(&HeaderMap::new());
        let tz = params_from_anthropic(params, &synapse).unwrap();
        assert_eq!(tz.model_name.as_deref(), Some("claude-sonnet-4-5"));
        assert_eq!(tz.params.chat_completion.thinking_budget_tokens, Some(1024));
        assert_eq!(tz.params.chat_completion.max_tokens, Some(32));
        assert!(tz.dynamic_tool_params.additional_tools.is_some());
    }

    #[gtest]
    fn lifts_system_role_from_messages() {
        let input = anthropic_to_input(
            None,
            vec![
                AnthropicIncomingMessage {
                    role: "system".to_string(),
                    content: AnthropicContent::Text("be terse".to_string()),
                },
                AnthropicIncomingMessage {
                    role: "user".to_string(),
                    content: AnthropicContent::Text("hi".to_string()),
                },
            ],
        )
        .expect("system-in-messages should be accepted");
        expect_that!(
            input.system,
            some(eq(&System::Text("be terse".to_string())))
        );
        expect_eq!(input.messages.len(), 1);
        expect_eq!(input.messages[0].role, Role::User);
    }

    #[gtest]
    fn concatenates_top_level_system_and_message_system() {
        let input = anthropic_to_input(
            Some(AnthropicSystem::Text("top".to_string())),
            vec![
                AnthropicIncomingMessage {
                    role: "system".to_string(),
                    content: AnthropicContent::Blocks(vec![AnthropicContentBlock::Text {
                        text: "from messages".to_string(),
                    }]),
                },
                AnthropicIncomingMessage {
                    role: "user".to_string(),
                    content: AnthropicContent::Text("hi".to_string()),
                },
            ],
        )
        .expect("combined system prompts should be accepted");
        expect_that!(
            input.system,
            some(eq(&System::Text("top\nfrom messages".to_string())))
        );
        expect_eq!(input.messages.len(), 1);
    }
}

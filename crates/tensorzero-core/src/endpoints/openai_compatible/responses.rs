// Modified by Delta-AI under Apache 2.0
//! OpenAI Responses API handler (`POST /v1/responses`).

use std::time::Instant;

use axum::Extension;
use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::Sse;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use tokio::time::sleep;
use tokio_stream::StreamExt;
use uuid::Uuid;

use crate::endpoints::inference::{InferenceOutput, InferenceResponseChunk, InferenceStream};
use crate::endpoints::openai_compatible::types::responses::{
    OpenAICompatibleResponsesParams, OpenAICompatibleResponsesResponse,
};
use crate::endpoints::openai_compatible::types::streaming::{
    SerializedSseEvent, value_to_sse_frame,
};
use crate::error::Error;
use crate::inference::types::{ContentBlockChunk, Usage, current_timestamp};
use crate::utils::gateway::AppState;
use tensorzero_auth::middleware::RequestApiKeyExtension;
use tensorzero_types::ApiType;

use super::infer::{error_response, infer_openai_compatible};
use super::stream_aggregator::{StreamAggregateRule, StreamAggregator};
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
        ApiType::Responses,
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
            let responses_stream = prepare_serialized_openai_responses_events(
                stream,
                inferred.response_model_prefix,
                inferred.synapse.stream_aggregate.clone(),
            )
            .map(|frame| frame.map(SerializedSseEvent::into_event));
            Sse::new(responses_stream)
                .keep_alive(axum::response::sse::KeepAlive::new())
                .into_response()
        }
    };
    inferred.synapse.apply_to_response(&mut response);
    Ok(response)
}

/// One frame of the Responses SSE lifecycle: an optional SSE event name plus
/// the JSON payload.
type ResponsesEventFrame = (Option<&'static str>, Value);

#[derive(Clone, Copy, PartialEq, Eq)]
enum OutputItemKind {
    Message,
    Reasoning,
}

/// The output item currently being streamed, if any.
struct OpenOutputItem {
    kind: OutputItemKind,
    item_id: String,
    output_index: usize,
    /// Accumulated message text or reasoning summary text.
    text: String,
    /// Reasoning items only: the provider's `encrypted_content`, replayed back
    /// by clients (e.g. the Vercel AI SDK) on subsequent agent-loop steps.
    signature: Option<String>,
}

/// A function call accumulated across the whole stream, keyed on its call id.
///
/// Providers emitting parallel tool calls (e.g. Volcengine Ark's Responses API
/// for DeepSeek models) do not guarantee that every lifecycle chunk for a call
/// arrives while that call is the "current" one: per-call `*.done` events can
/// be grouped after the following call's chunks already opened another item,
/// and argument deltas can interleave. Tracking calls by call id — the same
/// merge semantics the inference accumulator already applies when storing the
/// final output — keeps each call's name and arguments whole instead of
/// re-opening a name-less duplicate item for every out-of-order chunk.
struct FunctionCallAccumulator {
    output_index: usize,
    item_id: String,
    call_id: String,
    /// First non-empty name seen for this call id; argument-only chunks
    /// (deltas, `function_call_arguments.done`) carry none.
    name: String,
    arguments: String,
    /// Whether the `response.output_item.added` frame was emitted yet.
    added_emitted: bool,
}

/// Streaming state machine for the OpenAI Responses SSE lifecycle.
///
/// Converts TensorZero inference chunks into the full Responses event sequence
/// (`response.created` → `response.in_progress` → `response.output_item.added`
/// → `response.content_part.added` → deltas → `*.done` → `response.completed`),
/// including reasoning and function-call items, so clients like the Vercel AI
/// SDK that key off the `*.added` scaffolding events can parse the stream.
struct ResponsesStreamState {
    started: bool,
    response_id: String,
    inference_key: String,
    model: String,
    created_at: u64,
    next_output_index: usize,
    current: Option<OpenOutputItem>,
    /// Function calls, in first-seen call-id order; `function_call_index`
    /// maps call id → position so out-of-order chunks merge into the entry
    /// they belong to.
    function_calls: Vec<FunctionCallAccumulator>,
    function_call_index: std::collections::HashMap<String, usize>,
    completed_items: Vec<(usize, Value)>,
    last_usage: Option<Usage>,
    /// `encrypted_content` seen on a provider reasoning item before its text
    /// deltas open the synthesized item; applied when the item opens.
    pending_reasoning_signature: Option<String>,
}

impl ResponsesStreamState {
    fn new(inference_id: Uuid, variant_name: &str, response_model_prefix: &str) -> Self {
        Self {
            started: false,
            response_id: format!("resp_{inference_id}"),
            inference_key: inference_id.to_string(),
            model: format!("{response_model_prefix}{variant_name}"),
            created_at: current_timestamp(),
            next_output_index: 0,
            current: None,
            function_calls: Vec::new(),
            function_call_index: std::collections::HashMap::new(),
            completed_items: Vec::new(),
            last_usage: None,
            pending_reasoning_signature: None,
        }
    }

    fn started_frames(&self) -> Vec<ResponsesEventFrame> {
        let response = json!({
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": "in_progress",
            "model": self.model,
            "output": [],
        });
        vec![
            (
                Some("response.created"),
                json!({"type": "response.created", "response": response}),
            ),
            (
                Some("response.in_progress"),
                json!({"type": "response.in_progress", "response": response}),
            ),
        ]
    }

    /// Open a new output item of `kind` unless one is already open. Closes any
    /// previous item first. Function calls do not flow through here — they
    /// accumulate in [`Self::function_calls`] for the whole stream.
    fn ensure_item(
        &mut self,
        kind: OutputItemKind,
        frames: &mut Vec<ResponsesEventFrame>,
    ) {
        if matches!(&self.current, Some(item) if item.kind == kind) {
            return;
        }
        // Consumed by the Reasoning arm below and the OpenOutputItem at the
        // end of this function; None for message items.
        let signature = (kind == OutputItemKind::Reasoning)
            .then(|| self.pending_reasoning_signature.take())
            .flatten();
        self.close_current(frames);
        let output_index = self.next_output_index;
        self.next_output_index += 1;
        let item_id = match kind {
            OutputItemKind::Message => format!("msg_{}_{output_index}", self.inference_key),
            OutputItemKind::Reasoning => format!("rs_{}_{output_index}", self.inference_key),
        };
        match kind {
            OutputItemKind::Message => {
                frames.push((
                    Some("response.output_item.added"),
                    json!({
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": {
                            "id": item_id,
                            "type": "message",
                            "status": "in_progress",
                            "role": "assistant",
                            "content": [],
                        },
                    }),
                ));
                frames.push((
                    Some("response.content_part.added"),
                    json!({
                        "type": "response.content_part.added",
                        "item_id": item_id,
                        "output_index": output_index,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": []},
                    }),
                ));
            }
            OutputItemKind::Reasoning => {
                let mut item = json!({
                    "id": item_id,
                    "type": "reasoning",
                    "summary": [],
                    "encrypted_content": signature,
                });
                if signature.is_none()
                    && let Some(object) = item.as_object_mut()
                {
                    object.remove("encrypted_content");
                }
                frames.push((
                    Some("response.output_item.added"),
                    json!({
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": item,
                    }),
                ));
                frames.push((
                    Some("response.reasoning_summary_part.added"),
                    json!({
                        "type": "response.reasoning_summary_part.added",
                        "item_id": item_id,
                        "output_index": output_index,
                        "summary_index": 0,
                        "part": {"type": "summary_text", "text": ""},
                    }),
                ));
            }
        }
        self.current = Some(OpenOutputItem {
            kind,
            item_id,
            output_index,
            text: String::new(),
            signature,
        });
    }

    fn close_current(&mut self, frames: &mut Vec<ResponsesEventFrame>) {
        let Some(item) = self.current.take() else {
            return;
        };
        match item.kind {
            OutputItemKind::Message => {
                let part = json!({
                    "type": "output_text",
                    "text": item.text,
                    "annotations": [],
                });
                frames.push((
                    Some("response.output_text.done"),
                    json!({
                        "type": "response.output_text.done",
                        "item_id": item.item_id,
                        "output_index": item.output_index,
                        "content_index": 0,
                        "text": item.text,
                    }),
                ));
                frames.push((
                    Some("response.content_part.done"),
                    json!({
                        "type": "response.content_part.done",
                        "item_id": item.item_id,
                        "output_index": item.output_index,
                        "content_index": 0,
                        "part": part,
                    }),
                ));
                let message = json!({
                    "id": item.item_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [part],
                });
                frames.push((
                    Some("response.output_item.done"),
                    json!({
                        "type": "response.output_item.done",
                        "output_index": item.output_index,
                        "item": message,
                    }),
                ));
                self.completed_items.push((item.output_index, message));
            }
            OutputItemKind::Reasoning => {
                let part = json!({"type": "summary_text", "text": item.text});
                frames.push((
                    Some("response.reasoning_summary_text.done"),
                    json!({
                        "type": "response.reasoning_summary_text.done",
                        "item_id": item.item_id,
                        "output_index": item.output_index,
                        "summary_index": 0,
                        "text": item.text,
                    }),
                ));
                frames.push((
                    Some("response.reasoning_summary_part.done"),
                    json!({
                        "type": "response.reasoning_summary_part.done",
                        "item_id": item.item_id,
                        "output_index": item.output_index,
                        "summary_index": 0,
                        "part": part,
                    }),
                ));
                let mut reasoning = json!({
                    "id": item.item_id,
                    "type": "reasoning",
                    "summary": [part],
                    "encrypted_content": item.signature,
                });
                if let Some(object) = reasoning.as_object_mut() {
                    // Same text as the summary: DeepSeek thinking mode
                    // requires `content[].reasoning_text` on replay, and
                    // clients store this completed item as-is.
                    if !item.text.is_empty() {
                        object.insert(
                            "content".to_string(),
                            json!([{"type": "reasoning_text", "text": item.text}]),
                        );
                    }
                    if item.signature.is_none() {
                        object.remove("encrypted_content");
                    }
                }
                frames.push((
                    Some("response.output_item.done"),
                    json!({
                        "type": "response.output_item.done",
                        "output_index": item.output_index,
                        "item": reasoning,
                    }),
                ));
                self.completed_items.push((item.output_index, reasoning));
            }
        }
    }

    fn process_chunk(&mut self, chunk: InferenceResponseChunk) -> Vec<ResponsesEventFrame> {
        let mut frames = Vec::new();
        match chunk {
            InferenceResponseChunk::Chat(c) => {
                if let Some(usage) = c.usage {
                    self.last_usage = Some(usage);
                }
                for block in c.content {
                    match block {
                        ContentBlockChunk::Text(text) => {
                            if text.text.is_empty() {
                                continue;
                            }
                            self.ensure_item(OutputItemKind::Message, &mut frames);
                            let Some(item) = self.current.as_mut() else {
                                continue;
                            };
                            item.text.push_str(&text.text);
                            frames.push((
                                Some("response.output_text.delta"),
                                json!({
                                    "type": "response.output_text.delta",
                                    "item_id": item.item_id,
                                    "output_index": item.output_index,
                                    "content_index": 0,
                                    "delta": text.text,
                                }),
                            ));
                        }
                        ContentBlockChunk::Thought(thought) => {
                            let text = thought
                                .summary_text
                                .or(thought.text)
                                .filter(|value| !value.is_empty());
                            let Some(text) = text else {
                                continue;
                            };
                            self.ensure_item(OutputItemKind::Reasoning, &mut frames);
                            let Some(item) = self.current.as_mut() else {
                                continue;
                            };
                            item.text.push_str(&text);
                            frames.push((
                                Some("response.reasoning_summary_text.delta"),
                                json!({
                                    "type": "response.reasoning_summary_text.delta",
                                    "item_id": item.item_id,
                                    "output_index": item.output_index,
                                    "summary_index": 0,
                                    "delta": text,
                                }),
                            ));
                        }
                        ContentBlockChunk::ToolCall(tool_call) => {
                            let name = tool_call.raw_name.unwrap_or_default();
                            // Close any open message/reasoning item; function
                            // calls accumulate by call id and are closed only
                            // in `finish`, so a chunk for a different call
                            // never re-opens a duplicate item here.
                            self.close_current(&mut frames);
                            let call = self.function_call_entry(&tool_call.id);
                            if !name.is_empty() && call.name.is_empty() {
                                call.name = name;
                            }
                            if !call.added_emitted {
                                call.added_emitted = true;
                                frames.push((
                                    Some("response.output_item.added"),
                                    json!({
                                        "type": "response.output_item.added",
                                        "output_index": call.output_index,
                                        "item": {
                                            "id": call.item_id,
                                            "type": "function_call",
                                            "call_id": call.call_id,
                                            "name": call.name,
                                            "arguments": "",
                                            "status": "in_progress",
                                        },
                                    }),
                                ));
                            }
                            if tool_call.raw_arguments.is_empty() {
                                continue;
                            }
                            call.arguments.push_str(&tool_call.raw_arguments);
                            frames.push((
                                Some("response.function_call_arguments.delta"),
                                json!({
                                    "type": "response.function_call_arguments.delta",
                                    "item_id": call.item_id,
                                    "output_index": call.output_index,
                                    "delta": tool_call.raw_arguments,
                                }),
                            ));
                        }
                        ContentBlockChunk::Unknown(unknown) => {
                            // Provider reasoning items arrive as Unknown
                            // blocks carrying the raw item JSON. Harvest the
                            // `encrypted_content` so the synthesized reasoning
                            // item stays replayable: without it clients like
                            // the Vercel AI SDK cannot reconstruct a reasoning
                            // part for the next agent-loop step, and DeepSeek
                            // rejects the follow-up request for missing
                            // reasoning_text.
                            if unknown.data.get("type").and_then(Value::as_str) == Some("reasoning")
                                && let Some(encrypted) = unknown
                                    .data
                                    .get("encrypted_content")
                                    .and_then(Value::as_str)
                            {
                                match &mut self.current {
                                    Some(item) if item.kind == OutputItemKind::Reasoning => {
                                        item.signature.get_or_insert_with(|| encrypted.to_string());
                                    }
                                    _ => {
                                        self.pending_reasoning_signature =
                                            Some(encrypted.to_string());
                                    }
                                }
                            }
                        }
                    }
                }
            }
            InferenceResponseChunk::Json(c) => {
                if let Some(usage) = c.usage {
                    self.last_usage = Some(usage);
                }
                if !c.raw.is_empty() {
                    self.ensure_item(OutputItemKind::Message, &mut frames);
                    let Some(item) = self.current.as_mut() else {
                        return frames;
                    };
                    item.text.push_str(&c.raw);
                    frames.push((
                        Some("response.output_text.delta"),
                        json!({
                            "type": "response.output_text.delta",
                            "item_id": item.item_id,
                            "output_index": item.output_index,
                            "content_index": 0,
                            "delta": c.raw,
                        }),
                    ));
                }
            }
        }
        frames
    }

    /// Look up (or open) the accumulator for a call id. The `added` frame is
    /// emitted lazily by the caller on first sight so the item carries a name
    /// whenever the opening chunk provided one.
    fn function_call_entry(&mut self, call_id: &str) -> &mut FunctionCallAccumulator {
        let index = *self
            .function_call_index
            .entry(call_id.to_string())
            .or_insert_with(|| {
                let output_index = self.next_output_index;
                self.next_output_index += 1;
                self.function_calls.push(FunctionCallAccumulator {
                    output_index,
                    item_id: format!("fc_{call_id}"),
                    call_id: call_id.to_string(),
                    name: String::new(),
                    arguments: String::new(),
                    added_emitted: false,
                });
                self.function_calls.len() - 1
            });
        &mut self.function_calls[index]
    }

    fn finish(&mut self) -> Vec<ResponsesEventFrame> {
        let mut frames = Vec::new();
        // Function calls close only here, in first-seen order, each with the
        // name and arguments accumulated across the whole stream.
        for call in &self.function_calls {
            frames.push((
                Some("response.function_call_arguments.done"),
                json!({
                    "type": "response.function_call_arguments.done",
                    "item_id": call.item_id,
                    "output_index": call.output_index,
                    "arguments": call.arguments,
                }),
            ));
            let function_call = json!({
                "id": call.item_id,
                "type": "function_call",
                "call_id": call.call_id,
                "name": call.name,
                "arguments": call.arguments,
                "status": "completed",
            });
            frames.push((
                Some("response.output_item.done"),
                json!({
                    "type": "response.output_item.done",
                    "output_index": call.output_index,
                    "item": function_call,
                }),
            ));
            self.completed_items.push((call.output_index, function_call));
        }
        self.function_calls.clear();
        self.function_call_index.clear();
        self.close_current(&mut frames);
        // `current` closes mid-stream while function calls only close here, so
        // restore output-index order for the terminal `output` array.
        self.completed_items.sort_by_key(|(output_index, _)| *output_index);
        let usage = self.last_usage.take().unwrap_or_default();
        let output_items: Vec<Value> = self
            .completed_items
            .iter()
            .map(|(_, item)| item.clone())
            .collect();
        frames.push((
            Some("response.completed"),
            json!({
                "type": "response.completed",
                "response": {
                    "id": self.response_id,
                    "object": "response",
                    "created_at": self.created_at,
                    "status": "completed",
                    "model": self.model,
                    "output": output_items,
                    "usage": {
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "total_tokens": usage.total_tokens(),
                    },
                }
            }),
        ));
        frames
    }
}

/// Route outbound frames through the stream aggregator when
/// `x-synapse-stream-aggregate` / `x-tensorzero-stream-aggregate` rules are
/// active; otherwise serialize them directly.
fn aggregate_or_serialize(
    aggregator: &mut Option<StreamAggregator>,
    frames: Vec<ResponsesEventFrame>,
) -> Vec<Result<SerializedSseEvent, Error>> {
    let mut out = Vec::new();
    for (event, value) in frames {
        match aggregator {
            Some(agg) => {
                for (event, value) in agg.push(event, &value.to_string(), Instant::now()) {
                    out.push(Ok(value_to_sse_frame(event, value)));
                }
            }
            None => out.push(SerializedSseEvent::json(event.map(str::to_string), &value)),
        }
    }
    out
}

pub(super) fn prepare_serialized_openai_responses_events(
    mut stream: InferenceStream,
    response_model_prefix: String,
    stream_aggregate: Option<Vec<StreamAggregateRule>>,
) -> impl futures::Stream<Item = Result<SerializedSseEvent, Error>> {
    async_stream::stream! {
        let mut state: Option<ResponsesStreamState> = None;
        let mut aggregator = stream_aggregate.map(StreamAggregator::new);

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
                    let error_event = e.build_streaming_error_event(true, false);
                    for frame in aggregate_or_serialize(&mut aggregator, vec![(None, error_event)])
                    {
                        yield frame;
                    }
                    continue;
                }
            };

            let (inference_id, variant_name) = match &chunk {
                InferenceResponseChunk::Chat(c) => (c.inference_id, c.variant_name.as_str()),
                InferenceResponseChunk::Json(c) => (c.inference_id, c.variant_name.as_str()),
            };
            let state = state.get_or_insert_with(|| {
                ResponsesStreamState::new(inference_id, variant_name, &response_model_prefix)
            });
            if !state.started {
                for frame in
                    aggregate_or_serialize(&mut aggregator, state.started_frames())
                {
                    yield frame;
                }
                state.started = true;
            }

            let frames = state.process_chunk(chunk);
            for frame in aggregate_or_serialize(&mut aggregator, frames) {
                yield frame;
            }
        }

        if let Some(state) = state.as_mut() {
            for frame in aggregate_or_serialize(&mut aggregator, state.finish()) {
                yield frame;
            }
        }
        if let Some(agg) = aggregator.as_mut() {
            for (event, value) in agg.finish() {
                yield Ok(value_to_sse_frame(event, value));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::endpoints::inference::ChatInferenceResponseChunk;
    use crate::endpoints::openai_compatible::stream_aggregator::AggregatePart;
    use crate::inference::types::{TextChunk, ThoughtChunk, UnknownChunk};
    use crate::tool::ToolCallChunk;
    use googletest::matchers::{elements_are, eq};
    use googletest::{assert_that, expect_that, gtest};

    fn chat_chunk(
        inference_id: Uuid,
        content: Vec<ContentBlockChunk>,
        usage: Option<Usage>,
    ) -> InferenceResponseChunk {
        InferenceResponseChunk::Chat(ChatInferenceResponseChunk {
            inference_id,
            episode_id: Uuid::now_v7(),
            variant_name: "test_variant".to_string(),
            content,
            usage,
            raw_usage: None,
            raw_response: None,
            finish_reason: None,
            original_chunk: None,
            raw_chunk: None,
            aggregated_response: None,
        })
    }

    fn text_block(text: &str) -> ContentBlockChunk {
        ContentBlockChunk::Text(TextChunk {
            id: "0".to_string(),
            text: text.to_string(),
        })
    }

    fn tool_call_block(id: &str, name: Option<&str>, arguments: &str) -> ContentBlockChunk {
        ContentBlockChunk::ToolCall(ToolCallChunk {
            id: id.to_string(),
            raw_name: name.map(str::to_string),
            raw_arguments: arguments.to_string(),
        })
    }

    fn test_usage() -> Usage {
        Usage {
            input_tokens: Some(10),
            output_tokens: Some(5),
            ..Default::default()
        }
    }

    async fn collect_frames(
        chunks: Vec<InferenceResponseChunk>,
        stream_aggregate: Option<Vec<StreamAggregateRule>>,
    ) -> Vec<(Option<String>, Value)> {
        let stream: InferenceStream = Box::pin(futures::StreamExt::fuse(futures::stream::iter(
            chunks.into_iter().map(Ok),
        )));
        let events =
            futures::StreamExt::collect::<Vec<_>>(prepare_serialized_openai_responses_events(
                stream,
                "test-prefix/".to_string(),
                stream_aggregate,
            ))
            .await;
        events
            .into_iter()
            .map(|frame| {
                let frame = frame.expect("frame should serialize");
                let data = serde_json::from_str(&frame.data).expect("frame data should be JSON");
                (frame.event, data)
            })
            .collect()
    }

    fn event_names(frames: &[(Option<String>, Value)]) -> Vec<&str> {
        frames
            .iter()
            .map(|(event, _)| event.as_deref().expect("event should be named"))
            .collect()
    }

    #[gtest]
    #[tokio::test]
    async fn test_streaming_emits_full_responses_lifecycle() {
        let inference_id = Uuid::now_v7();
        let frames = collect_frames(
            vec![
                chat_chunk(inference_id, vec![text_block("Hello")], None),
                chat_chunk(inference_id, vec![text_block(" world")], None),
                chat_chunk(
                    inference_id,
                    vec![tool_call_block(
                        "call_1",
                        Some("get_weather"),
                        "{\"location\":",
                    )],
                    None,
                ),
                chat_chunk(
                    inference_id,
                    vec![tool_call_block("call_1", None, "\"Paris\"}")],
                    None,
                ),
                chat_chunk(inference_id, vec![], Some(test_usage())),
            ],
            None,
        )
        .await;

        assert_that!(
            event_names(&frames),
            elements_are![
                eq(&"response.created"),
                eq(&"response.in_progress"),
                eq(&"response.output_item.added"),
                eq(&"response.content_part.added"),
                eq(&"response.output_text.delta"),
                eq(&"response.output_text.delta"),
                eq(&"response.output_text.done"),
                eq(&"response.content_part.done"),
                eq(&"response.output_item.done"),
                eq(&"response.output_item.added"),
                eq(&"response.function_call_arguments.delta"),
                eq(&"response.function_call_arguments.delta"),
                eq(&"response.function_call_arguments.done"),
                eq(&"response.output_item.done"),
                eq(&"response.completed"),
            ]
        );

        // Scaffolding for the message item.
        let added = &frames[2].1;
        expect_that!(added["item"]["type"].as_str(), eq(Some("message")));
        expect_that!(added["item"]["status"].as_str(), eq(Some("in_progress")));
        expect_that!(added["output_index"].as_u64(), eq(Some(0)));

        // Text deltas then done with the full text.
        expect_that!(frames[4].1["delta"].as_str(), eq(Some("Hello")));
        expect_that!(frames[5].1["delta"].as_str(), eq(Some(" world")));
        expect_that!(frames[6].1["text"].as_str(), eq(Some("Hello world")));
        expect_that!(frames[6].1["item_id"], eq(&frames[2].1["item"]["id"]));

        // Function-call item.
        let call_added = &frames[9].1;
        expect_that!(
            call_added["item"]["type"].as_str(),
            eq(Some("function_call"))
        );
        expect_that!(call_added["item"]["call_id"].as_str(), eq(Some("call_1")));
        expect_that!(call_added["item"]["name"].as_str(), eq(Some("get_weather")));
        expect_that!(call_added["output_index"].as_u64(), eq(Some(1)));
        expect_that!(
            frames[12].1["arguments"].as_str(),
            eq(Some("{\"location\":\"Paris\"}"))
        );

        // Terminal event carries the completed output items and usage.
        let completed = &frames[14].1["response"];
        expect_that!(completed["status"].as_str(), eq(Some("completed")));
        let output = completed["output"].as_array().expect("output items");
        assert_that!(output.len(), eq(2));
        expect_that!(output[0]["type"].as_str(), eq(Some("message")));
        expect_that!(
            output[0]["content"][0]["text"].as_str(),
            eq(Some("Hello world"))
        );
        expect_that!(output[1]["type"].as_str(), eq(Some("function_call")));
        expect_that!(output[1]["name"].as_str(), eq(Some("get_weather")));
        expect_that!(
            output[1]["arguments"].as_str(),
            eq(Some("{\"location\":\"Paris\"}"))
        );
        expect_that!(completed["usage"]["input_tokens"].as_u64(), eq(Some(10)));
        expect_that!(completed["usage"]["output_tokens"].as_u64(), eq(Some(5)));
    }

    /// Parallel tool calls whose per-call lifecycle chunks arrive grouped
    /// instead of nested: both `added` chunks first, then both calls'
    /// argument deltas, then both name-less `done` chunks (the shape Volcengine
    /// Ark's Responses API produces for DeepSeek parallel tool calls). The
    /// stream must still carry exactly one item per call, each with its name
    /// and full arguments — previously every out-of-order chunk re-opened a
    /// duplicate `function_call` item with an empty name, which clients
    /// replayed back and providers rejected with `400 MissingParameter:
    /// input.name`.
    #[gtest]
    #[tokio::test]
    async fn test_streaming_grouped_parallel_tool_call_chunks() {
        let inference_id = Uuid::now_v7();
        let frames = collect_frames(
            vec![
                chat_chunk(
                    inference_id,
                    vec![tool_call_block("call_a", Some("search"), "{\"q\":")],
                    None,
                ),
                chat_chunk(
                    inference_id,
                    vec![tool_call_block("call_b", Some("search"), "{\"q\":")],
                    None,
                ),
                chat_chunk(inference_id, vec![tool_call_block("call_a", None, "\"a\"}")], None),
                chat_chunk(inference_id, vec![tool_call_block("call_b", None, "\"b\"}")], None),
                // Late name-less done chunks for both calls, after the other
                // call's item already opened.
                chat_chunk(inference_id, vec![tool_call_block("call_a", None, "")], None),
                chat_chunk(inference_id, vec![tool_call_block("call_b", None, "")], None),
                chat_chunk(inference_id, vec![], Some(test_usage())),
            ],
            None,
        )
        .await;

        let added: Vec<&Value> = frames
            .iter()
            .filter(|(event, value)| {
                event.as_deref() == Some("response.output_item.added")
                    && value["item"]["type"] == "function_call"
            })
            .map(|(_, value)| value)
            .collect();
        assert_eq!(added.len(), 2, "one item per call, no duplicates");
        expect_that!(added[0]["item"]["name"].as_str(), eq(Some("search")));
        expect_that!(added[0]["item"]["call_id"].as_str(), eq(Some("call_a")));
        expect_that!(added[1]["item"]["name"].as_str(), eq(Some("search")));
        expect_that!(added[1]["item"]["call_id"].as_str(), eq(Some("call_b")));

        let done: Vec<&Value> = frames
            .iter()
            .filter(|(event, value)| {
                event.as_deref() == Some("response.output_item.done")
                    && value["item"]["type"] == "function_call"
            })
            .map(|(_, value)| value)
            .collect();
        assert_eq!(done.len(), 2);
        expect_that!(done[0]["item"]["call_id"].as_str(), eq(Some("call_a")));
        expect_that!(
            done[0]["item"]["arguments"].as_str(),
            eq(Some("{\"q\":\"a\"}"))
        );
        expect_that!(done[1]["item"]["call_id"].as_str(), eq(Some("call_b")));
        expect_that!(
            done[1]["item"]["arguments"].as_str(),
            eq(Some("{\"q\":\"b\"}"))
        );
        assert!(
            done.iter().all(|frame| frame["item"]["name"] == "search"),
            "no name-less duplicate items may be synthesized"
        );

        let completed = frames
            .iter()
            .find(|(event, _)| event.as_deref() == Some("response.completed"))
            .expect("completed frame");
        let output = completed.1["response"]["output"].as_array().expect("output");
        assert_that!(output.len(), eq(2));
        expect_that!(output[0]["call_id"].as_str(), eq(Some("call_a")));
        expect_that!(output[0]["name"].as_str(), eq(Some("search")));
        expect_that!(output[1]["call_id"].as_str(), eq(Some("call_b")));
        expect_that!(output[1]["name"].as_str(), eq(Some("search")));
    }

    /// Argument deltas interleaved across two open calls must each accumulate
    /// onto their own call, in arrival order.
    #[gtest]
    #[tokio::test]
    async fn test_streaming_interleaved_parallel_tool_call_deltas() {
        let inference_id = Uuid::now_v7();
        let frames = collect_frames(
            vec![
                chat_chunk(
                    inference_id,
                    vec![tool_call_block("call_a", Some("search"), "")],
                    None,
                ),
                chat_chunk(
                    inference_id,
                    vec![tool_call_block("call_b", Some("fetch"), "")],
                    None,
                ),
                chat_chunk(inference_id, vec![tool_call_block("call_b", None, "{\"b\"")], None),
                chat_chunk(inference_id, vec![tool_call_block("call_a", None, "{\"a\"")], None),
                chat_chunk(inference_id, vec![tool_call_block("call_b", None, ":1}")], None),
                chat_chunk(inference_id, vec![tool_call_block("call_a", None, ":2}")], None),
                chat_chunk(inference_id, vec![], Some(test_usage())),
            ],
            None,
        )
        .await;

        let completed = frames
            .iter()
            .find(|(event, _)| event.as_deref() == Some("response.completed"))
            .expect("completed frame");
        let output = completed.1["response"]["output"].as_array().expect("output");
        assert_that!(output.len(), eq(2));
        expect_that!(output[0]["call_id"].as_str(), eq(Some("call_a")));
        expect_that!(output[0]["name"].as_str(), eq(Some("search")));
        expect_that!(
            output[0]["arguments"].as_str(),
            eq(Some("{\"a\":2}"))
        );
        expect_that!(output[1]["call_id"].as_str(), eq(Some("call_b")));
        expect_that!(output[1]["name"].as_str(), eq(Some("fetch")));
        expect_that!(
            output[1]["arguments"].as_str(),
            eq(Some("{\"b\":1}"))
        );
    }

    #[gtest]
    #[tokio::test]
    async fn test_streaming_emits_reasoning_item_events() {
        let inference_id = Uuid::now_v7();
        let thought = ContentBlockChunk::Thought(ThoughtChunk {
            id: "0".to_string(),
            text: None,
            signature: None,
            summary_id: None,
            summary_text: Some("thinking hard".to_string()),
            provider_type: None,
            extra_data: None,
        });
        let frames = collect_frames(
            vec![
                chat_chunk(inference_id, vec![thought], None),
                chat_chunk(inference_id, vec![text_block("answer")], Some(test_usage())),
            ],
            None,
        )
        .await;

        assert_that!(
            event_names(&frames),
            elements_are![
                eq(&"response.created"),
                eq(&"response.in_progress"),
                eq(&"response.output_item.added"),
                eq(&"response.reasoning_summary_part.added"),
                eq(&"response.reasoning_summary_text.delta"),
                eq(&"response.reasoning_summary_text.done"),
                eq(&"response.reasoning_summary_part.done"),
                eq(&"response.output_item.done"),
                eq(&"response.output_item.added"),
                eq(&"response.content_part.added"),
                eq(&"response.output_text.delta"),
                eq(&"response.output_text.done"),
                eq(&"response.content_part.done"),
                eq(&"response.output_item.done"),
                eq(&"response.completed"),
            ]
        );
        expect_that!(frames[2].1["item"]["type"].as_str(), eq(Some("reasoning")));
        expect_that!(frames[4].1["delta"].as_str(), eq(Some("thinking hard")));
        let output = frames[14].1["response"]["output"]
            .as_array()
            .expect("output items");
        assert_that!(output.len(), eq(2));
        expect_that!(output[0]["type"].as_str(), eq(Some("reasoning")));
        expect_that!(
            output[0]["summary"][0]["text"].as_str(),
            eq(Some("thinking hard"))
        );
        expect_that!(
            output[0]["content"][0]["type"].as_str(),
            eq(Some("reasoning_text"))
        );
        expect_that!(
            output[0]["content"][0]["text"].as_str(),
            eq(Some("thinking hard"))
        );
    }

    #[gtest]
    #[tokio::test]
    async fn test_streaming_reasoning_item_carries_encrypted_content() {
        // Providers forward reasoning output items as Unknown chunks carrying
        // the raw item JSON. The synthesized Responses events must include the
        // item's `encrypted_content` — without it, clients like the Vercel AI
        // SDK cannot build a replayable reasoning part, and DeepSeek rejects
        // the next agent-loop step for missing reasoning_text.
        let inference_id = Uuid::now_v7();
        let provider_reasoning_item = ContentBlockChunk::Unknown(UnknownChunk {
            id: "0".to_string(),
            data: serde_json::json!({
                "type": "reasoning",
                "id": "rs_provider_1",
                "status": "in_progress",
                "content": [],
                "summary": [],
                "encrypted_content": "enc-payload",
            }),
            model_name: Some("deepseek-v4-flash".to_string()),
            provider_name: Some("deepseek".to_string()),
        });
        let thought = ContentBlockChunk::Thought(ThoughtChunk {
            id: "0".to_string(),
            text: None,
            signature: None,
            summary_id: None,
            summary_text: Some("thinking hard".to_string()),
            provider_type: None,
            extra_data: None,
        });
        let frames = collect_frames(
            vec![
                chat_chunk(inference_id, vec![provider_reasoning_item], None),
                chat_chunk(inference_id, vec![thought], None),
                chat_chunk(inference_id, vec![text_block("answer")], Some(test_usage())),
            ],
            None,
        )
        .await;

        let added = frames
            .iter()
            .find(|(event, value)| {
                event.as_deref() == Some("response.output_item.added")
                    && value["item"]["type"] == "reasoning"
            })
            .expect("reasoning output_item.added frame");
        expect_that!(
            added.1["item"]["encrypted_content"].as_str(),
            eq(Some("enc-payload"))
        );

        let done = frames
            .iter()
            .find(|(event, value)| {
                event.as_deref() == Some("response.output_item.done")
                    && value["item"]["type"] == "reasoning"
            })
            .expect("reasoning output_item.done frame");
        expect_that!(
            done.1["item"]["encrypted_content"].as_str(),
            eq(Some("enc-payload"))
        );

        let completed = frames
            .iter()
            .find(|(event, _)| event.as_deref() == Some("response.completed"))
            .expect("response.completed frame");
        let output = completed.1["response"]["output"]
            .as_array()
            .expect("output items");
        let reasoning = output
            .iter()
            .find(|item| item["type"] == "reasoning")
            .expect("reasoning output item");
        expect_that!(
            reasoning["encrypted_content"].as_str(),
            eq(Some("enc-payload"))
        );
        expect_that!(
            reasoning["summary"][0]["text"].as_str(),
            eq(Some("thinking hard"))
        );
        expect_that!(
            reasoning["content"][0]["type"].as_str(),
            eq(Some("reasoning_text"))
        );
        expect_that!(
            reasoning["content"][0]["text"].as_str(),
            eq(Some("thinking hard"))
        );
    }

    #[gtest]
    #[tokio::test]
    async fn test_stream_aggregate_merges_responses_text_deltas() {
        let inference_id = Uuid::now_v7();
        let rules = vec![StreamAggregateRule {
            part: AggregatePart::Content,
            start_delay_ms: 0,
            interval_ms: 10_000,
            max_chars: 500,
        }];
        let frames = collect_frames(
            vec![
                chat_chunk(inference_id, vec![text_block("Hel")], None),
                chat_chunk(inference_id, vec![text_block("lo")], None),
                chat_chunk(inference_id, vec![text_block("!")], None),
                chat_chunk(inference_id, vec![], Some(test_usage())),
            ],
            Some(rules),
        )
        .await;

        let deltas: Vec<&Value> = frames
            .iter()
            .filter(|(event, _)| event.as_deref() == Some("response.output_text.delta"))
            .map(|(_, data)| data)
            .collect();
        assert_that!(deltas.len(), eq(1));
        expect_that!(deltas[0]["delta"].as_str(), eq(Some("Hello!")));

        // The lifecycle events are unaffected by aggregation.
        let names = event_names(&frames);
        expect_that!(names.first(), eq(Some(&"response.created")));
        expect_that!(names.last(), eq(Some(&"response.completed")));
        let completed = &frames.last().expect("completed frame").1["response"];
        expect_that!(
            completed["output"][0]["content"][0]["text"].as_str(),
            eq(Some("Hello!"))
        );
    }
}

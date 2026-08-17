// Modified by Delta-AI under Apache 2.0
//! OpenAI Responses API (`POST /v1/responses`) types.
//!
//! This is an inbound adapter: we convert Responses `input` / `instructions`
//! into TensorZero chat inference and map the result back to a Responses
//! object. It is not an upstream pass-through.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use uuid::Uuid;

use crate::cache::CacheParamsOptions;
use crate::config::Namespace;
use crate::endpoints::inference::{InferenceCredentials, InferenceParams, InferenceResponse};
use crate::endpoints::openai_compatible::types::chat_completions::{
    OpenAICompatibleAssistantMessage, OpenAICompatibleMessage, OpenAICompatibleParams,
    OpenAICompatibleStreamOptions, OpenAICompatibleSystemMessage, OpenAICompatibleUserMessage,
    process_chat_content,
};
use crate::endpoints::openai_compatible::types::tool::{
    ChatCompletionToolChoiceOption, OpenAICompatibleTool, OpenAICompatibleToolCall,
};
use crate::error::{Error, ErrorDetails};
use crate::inference::types::current_timestamp;

#[derive(Clone, Debug, Deserialize)]
pub struct OpenAICompatibleResponsesParams {
    pub model: String,
    pub input: Value,
    pub instructions: Option<String>,
    pub stream: Option<bool>,
    pub temperature: Option<f32>,
    pub max_output_tokens: Option<u32>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub seed: Option<u32>,
    pub tools: Option<Vec<OpenAICompatibleTool>>,
    pub tool_choice: Option<ChatCompletionToolChoiceOption>,
    pub parallel_tool_calls: Option<bool>,
    pub stream_options: Option<OpenAICompatibleStreamOptions>,
    #[serde(rename = "tensorzero::dryrun")]
    pub tensorzero_dryrun: Option<bool>,
    #[serde(rename = "tensorzero::episode_id")]
    pub tensorzero_episode_id: Option<Uuid>,
    #[serde(rename = "tensorzero::namespace")]
    pub tensorzero_namespace: Option<Namespace>,
    #[serde(rename = "tensorzero::cache_options")]
    pub tensorzero_cache_options: Option<CacheParamsOptions>,
    #[serde(default, rename = "tensorzero::credentials")]
    pub tensorzero_credentials: InferenceCredentials,
    #[serde(default, rename = "tensorzero::params")]
    pub tensorzero_params: Option<InferenceParams>,
    #[serde(default, rename = "tensorzero::include_raw_usage")]
    pub tensorzero_include_raw_usage: bool,
    #[serde(default, rename = "tensorzero::include_original_response")]
    pub tensorzero_include_original_response: bool,
    #[serde(default, rename = "tensorzero::include_raw_response")]
    pub tensorzero_include_raw_response: bool,
    #[serde(flatten)]
    pub unknown_fields: HashMap<String, Value>,
}

impl OpenAICompatibleResponsesParams {
    pub fn into_chat_params(self) -> Result<OpenAICompatibleParams, Error> {
        let messages = responses_input_to_messages(self.input, self.instructions)?;
        let max_tokens = match (self.max_output_tokens, self.max_tokens) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) | (None, Some(a)) => Some(a),
            (None, None) => None,
        };
        Ok(OpenAICompatibleParams {
            messages,
            model: self.model,
            frequency_penalty: self.frequency_penalty,
            max_tokens,
            presence_penalty: self.presence_penalty,
            seed: self.seed,
            stream: self.stream,
            stream_options: self.stream_options,
            temperature: self.temperature,
            top_p: self.top_p,
            tools: self.tools,
            tool_choice: self.tool_choice,
            parallel_tool_calls: self.parallel_tool_calls,
            tensorzero_dryrun: self.tensorzero_dryrun,
            tensorzero_episode_id: self.tensorzero_episode_id,
            tensorzero_namespace: self.tensorzero_namespace,
            tensorzero_cache_options: self.tensorzero_cache_options,
            tensorzero_credentials: self.tensorzero_credentials,
            tensorzero_params: self.tensorzero_params,
            tensorzero_include_raw_usage: self.tensorzero_include_raw_usage,
            tensorzero_include_original_response: self.tensorzero_include_original_response,
            tensorzero_include_raw_response: self.tensorzero_include_raw_response,
            unknown_fields: self.unknown_fields,
            ..Default::default()
        })
    }
}

pub fn responses_input_to_messages(
    input: Value,
    instructions: Option<String>,
) -> Result<Vec<OpenAICompatibleMessage>, Error> {
    let mut messages = Vec::new();
    if let Some(instructions) = instructions.filter(|value| !value.is_empty()) {
        messages.push(OpenAICompatibleMessage::System(
            OpenAICompatibleSystemMessage {
                content: Value::String(instructions),
            },
        ));
    }
    match input {
        Value::String(text) => {
            messages.push(user_message(Value::String(text)));
        }
        Value::Array(items) => {
            for item in items {
                messages.push(parse_responses_input_item(item)?);
            }
        }
        other => {
            return Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                message: format!(
                    "`input` must be a string or array, got {}",
                    value_kind(&other)
                ),
            }));
        }
    }
    if messages.is_empty() {
        return Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
            message: "`input` must not be empty".to_string(),
        }));
    }
    Ok(messages)
}

fn parse_responses_input_item(item: Value) -> Result<OpenAICompatibleMessage, Error> {
    if let Some(text) = item.as_str() {
        return Ok(user_message(Value::String(text.to_string())));
    }
    let obj = item.as_object().ok_or_else(|| {
        Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
            message: "`input` array items must be strings or objects".to_string(),
        })
    })?;
    let role = obj
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or("user")
        .to_string();
    let content = normalize_responses_content(obj.get("content").cloned().unwrap_or(Value::Null));
    match role.as_str() {
        "system" | "developer" => Ok(OpenAICompatibleMessage::System(
            OpenAICompatibleSystemMessage { content },
        )),
        "assistant" => Ok(OpenAICompatibleMessage::Assistant(
            OpenAICompatibleAssistantMessage {
                content: Some(content),
                tool_calls: None,
                tensorzero_extra_content: None,
            },
        )),
        _ => Ok(user_message(content)),
    }
}

fn normalize_responses_content(content: Value) -> Value {
    match content {
        Value::Null => Value::String(String::new()),
        Value::Array(parts) => {
            let converted = parts
                .into_iter()
                .map(|part| {
                    let part_type = part.get("type").and_then(Value::as_str).unwrap_or("");
                    if part_type == "input_text" || part_type == "output_text" {
                        json!({
                            "type": "text",
                            "text": part.get("text").cloned().unwrap_or(Value::String(String::new())),
                        })
                    } else {
                        part
                    }
                })
                .collect();
            Value::Array(converted)
        }
        other => other,
    }
}

fn user_message(content: Value) -> OpenAICompatibleMessage {
    OpenAICompatibleMessage::User(OpenAICompatibleUserMessage { content })
}

fn value_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct OpenAICompatibleResponsesUsage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct OpenAICompatibleResponsesResponse {
    pub id: String,
    pub object: String,
    pub created_at: u64,
    pub status: String,
    pub model: String,
    pub output: Vec<Value>,
    pub usage: OpenAICompatibleResponsesUsage,
}

impl From<(InferenceResponse, String)> for OpenAICompatibleResponsesResponse {
    fn from((inference_response, response_model_prefix): (InferenceResponse, String)) -> Self {
        match inference_response {
            InferenceResponse::Chat(response) => {
                let (content, tool_calls, _extra) = process_chat_content(response.content);
                let model = format!("{response_model_prefix}{}", response.variant_name);
                let usage = OpenAICompatibleResponsesUsage {
                    input_tokens: response.usage.input_tokens,
                    output_tokens: response.usage.output_tokens,
                    total_tokens: response.usage.total_tokens(),
                };
                OpenAICompatibleResponsesResponse {
                    id: format!("resp_{}", response.inference_id),
                    object: "response".to_string(),
                    created_at: current_timestamp(),
                    status: "completed".to_string(),
                    model,
                    output: responses_output_items(
                        &format!("msg_{}", response.inference_id),
                        content.as_deref(),
                        &tool_calls,
                    ),
                    usage,
                }
            }
            InferenceResponse::Json(response) => {
                let model = format!("{response_model_prefix}{}", response.variant_name);
                let usage = OpenAICompatibleResponsesUsage {
                    input_tokens: response.usage.input_tokens,
                    output_tokens: response.usage.output_tokens,
                    total_tokens: response.usage.total_tokens(),
                };
                OpenAICompatibleResponsesResponse {
                    id: format!("resp_{}", response.inference_id),
                    object: "response".to_string(),
                    created_at: current_timestamp(),
                    status: "completed".to_string(),
                    model,
                    output: responses_output_items(
                        &format!("msg_{}", response.inference_id),
                        response.output.raw.as_deref(),
                        &[],
                    ),
                    usage,
                }
            }
        }
    }
}

pub fn responses_output_items(
    message_id: &str,
    text: Option<&str>,
    tool_calls: &[OpenAICompatibleToolCall],
) -> Vec<Value> {
    let mut output = Vec::new();
    if let Some(text) = text.filter(|value| !value.is_empty()) {
        output.push(json!({
            "id": message_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": text,
                "annotations": []
            }]
        }));
    }
    for tool_call in tool_calls {
        output.push(json!({
            "type": "function_call",
            "id": tool_call.id,
            "call_id": tool_call.id,
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
            "status": "completed",
        }));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_input_with_instructions() {
        let messages =
            responses_input_to_messages(json!("Hello"), Some("You are helpful".into())).unwrap();
        assert_eq!(messages.len(), 2);
        match &messages[0] {
            OpenAICompatibleMessage::System(msg) => {
                assert_eq!(msg.content, Value::String("You are helpful".into()));
            }
            _ => panic!("expected system"),
        }
        match &messages[1] {
            OpenAICompatibleMessage::User(msg) => {
                assert_eq!(msg.content, Value::String("Hello".into()));
            }
            _ => panic!("expected user"),
        }
    }

    #[test]
    fn test_message_array_with_input_text_parts() {
        let messages = responses_input_to_messages(
            json!([
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hi"}]
                }
            ]),
            None,
        )
        .unwrap();
        match &messages[0] {
            OpenAICompatibleMessage::User(msg) => {
                assert_eq!(msg.content, json!([{"type": "text", "text": "Hi"}]));
            }
            _ => panic!("expected user"),
        }
    }

    #[test]
    fn test_empty_input_errors() {
        let err = responses_input_to_messages(json!([]), None).unwrap_err();
        assert!(err.to_string().contains("`input` must not be empty"));
    }
}

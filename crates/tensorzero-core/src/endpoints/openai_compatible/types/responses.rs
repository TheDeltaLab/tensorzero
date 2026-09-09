// Modified by Delta-AI under Apache 2.0
//! OpenAI Responses API (`POST /v1/responses`) types.
//!
//! This is an inbound adapter: we convert Responses `input` / `instructions`
//! into TensorZero chat inference and map the result back to a Responses
//! object. It is not an upstream pass-through.

use serde::{Deserialize, Serialize, de::Error as _};
use serde_json::{Value, json};
use std::collections::HashMap;
use uuid::Uuid;

use crate::cache::CacheParamsOptions;
use crate::config::Namespace;
use crate::endpoints::inference::{InferenceCredentials, InferenceParams, InferenceResponse};
use crate::endpoints::openai_compatible::types::chat_completions::{
    JsonSchemaInfo, OpenAICompatibleAssistantMessage, OpenAICompatibleMessage,
    OpenAICompatibleParams, OpenAICompatibleResponseFormat, OpenAICompatibleStreamOptions,
    OpenAICompatibleSystemMessage, OpenAICompatibleUserMessage, process_chat_content,
};
use crate::endpoints::openai_compatible::types::tool::{
    ChatCompletionToolChoiceOption, OpenAICompatibleFunctionTool, OpenAICompatibleTool,
    OpenAICompatibleToolCall,
};
use crate::error::{Error, ErrorDetails};
use crate::inference::types::chat_completion_inference_params::ServiceTier;
use crate::inference::types::current_timestamp;
use crate::tool::OpenAICustomTool;

/// Tool definition accepted by the OpenAI Responses API adapter.
///
/// The Responses API uses a flat tool shape
/// (`{"type": "function", "name": ..., "description": ..., "parameters": ..., "strict": ...}`),
/// unlike chat completions which wraps the definition in a `function` object. Both shapes are
/// accepted here so callers of either API style work against `/openai/v1/responses`.
#[derive(Clone, Debug, PartialEq)]
pub enum OpenAICompatibleResponsesTool {
    Flat(OpenAICompatibleResponsesFlatTool),
    Chat(OpenAICompatibleTool),
}

/// Flat Responses-API tool shape, tagged on `type`.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAICompatibleResponsesFlatTool {
    Function(OpenAICompatibleResponsesFunctionTool),
    Custom(OpenAICustomTool),
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct OpenAICompatibleResponsesFunctionTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<Value>,
    #[serde(default)]
    pub strict: Option<bool>,
}

impl<'de> Deserialize<'de> for OpenAICompatibleResponsesTool {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        // The chat-completions shape wraps the definition under `function` / `custom`;
        // the Responses-API shape is flat. Pick the parser based on which is present.
        if value.get("function").is_some() || value.get("custom").is_some() {
            let tool: OpenAICompatibleTool =
                serde_json::from_value(value).map_err(D::Error::custom)?;
            Ok(OpenAICompatibleResponsesTool::Chat(tool))
        } else {
            let tool: OpenAICompatibleResponsesFlatTool =
                serde_json::from_value(value).map_err(D::Error::custom)?;
            Ok(OpenAICompatibleResponsesTool::Flat(tool))
        }
    }
}

impl From<OpenAICompatibleResponsesTool> for OpenAICompatibleTool {
    fn from(tool: OpenAICompatibleResponsesTool) -> Self {
        match tool {
            OpenAICompatibleResponsesTool::Chat(tool) => tool,
            OpenAICompatibleResponsesTool::Flat(OpenAICompatibleResponsesFlatTool::Function(
                function,
            )) => OpenAICompatibleTool::Function {
                function: OpenAICompatibleFunctionTool {
                    name: function.name,
                    description: function.description,
                    parameters: function.parameters.unwrap_or(Value::Null),
                    strict: function.strict.unwrap_or(false),
                },
            },
            OpenAICompatibleResponsesTool::Flat(OpenAICompatibleResponsesFlatTool::Custom(
                custom,
            )) => OpenAICompatibleTool::Custom { custom },
        }
    }
}

/// `text` config accepted by the OpenAI Responses API adapter.
///
/// The Responses API nests the structured-output format under `text.format`
/// (flat shape) and the output verbosity under `text.verbosity`, unlike chat
/// completions which uses a top-level `response_format` with a nested
/// `json_schema` object.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct OpenAICompatibleResponsesText {
    #[serde(default)]
    pub format: Option<OpenAICompatibleResponsesTextFormat>,
    #[serde(default)]
    pub verbosity: Option<String>,
}

/// Flat Responses-API text format shape, tagged on `type`.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAICompatibleResponsesTextFormat {
    Text,
    JsonObject,
    JsonSchema {
        name: String,
        #[serde(default)]
        description: Option<String>,
        #[serde(default)]
        schema: Option<Value>,
        #[serde(default)]
        strict: Option<bool>,
    },
}

impl From<OpenAICompatibleResponsesTextFormat> for OpenAICompatibleResponseFormat {
    fn from(format: OpenAICompatibleResponsesTextFormat) -> Self {
        match format {
            OpenAICompatibleResponsesTextFormat::Text => OpenAICompatibleResponseFormat::Text,
            OpenAICompatibleResponsesTextFormat::JsonObject => {
                OpenAICompatibleResponseFormat::JsonObject
            }
            OpenAICompatibleResponsesTextFormat::JsonSchema {
                name,
                description,
                schema,
                strict,
            } => OpenAICompatibleResponseFormat::JsonSchema {
                json_schema: JsonSchemaInfo {
                    name,
                    description,
                    schema,
                    strict: strict.unwrap_or(false),
                },
            },
        }
    }
}

/// `reasoning` config accepted by the OpenAI Responses API adapter.
/// Keys other than `effort` (e.g. `summary`) are ignored.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct OpenAICompatibleResponsesReasoning {
    #[serde(default)]
    pub effort: Option<String>,
}

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
    pub tools: Option<Vec<OpenAICompatibleResponsesTool>>,
    pub tool_choice: Option<ChatCompletionToolChoiceOption>,
    pub parallel_tool_calls: Option<bool>,
    pub stream_options: Option<OpenAICompatibleStreamOptions>,
    pub text: Option<OpenAICompatibleResponsesText>,
    pub reasoning: Option<OpenAICompatibleResponsesReasoning>,
    pub service_tier: Option<ServiceTier>,
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
        let (response_format, verbosity) = match self.text {
            Some(text) => (text.format.map(Into::into), text.verbosity),
            None => (None, None),
        };
        let reasoning_effort = self.reasoning.and_then(|reasoning| reasoning.effort);
        Ok(OpenAICompatibleParams {
            messages,
            model: self.model,
            frequency_penalty: self.frequency_penalty,
            max_tokens,
            presence_penalty: self.presence_penalty,
            response_format,
            seed: self.seed,
            stream: self.stream,
            stream_options: self.stream_options,
            temperature: self.temperature,
            top_p: self.top_p,
            tools: self
                .tools
                .map(|tools| tools.into_iter().map(Into::into).collect()),
            tool_choice: self.tool_choice,
            parallel_tool_calls: self.parallel_tool_calls,
            reasoning_effort,
            service_tier: self.service_tier,
            verbosity,
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
    use googletest::prelude::*;

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

    #[gtest]
    fn test_flat_responses_function_tool_deserializes() {
        // Regression test: the OpenAI Responses API sends tools in a flat shape
        // (`{"type": "function", "name": ...}`), which previously failed with
        // `tools[0]: missing field 'function'`.
        let body = json!({
            "model": "gpt-5",
            "input": "What's the weather in Paris?",
            "tools": [{
                "type": "function",
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                },
                "strict": true
            }],
            "tool_choice": "auto"
        });
        let params: OpenAICompatibleResponsesParams =
            serde_json::from_value(body).expect("flat Responses tool body should deserialize");
        let tools = params.tools.as_deref().expect("tools should be present");
        let [tool] = tools else {
            panic!("expected exactly one tool, got {tools:?}");
        };
        let OpenAICompatibleResponsesTool::Flat(OpenAICompatibleResponsesFlatTool::Function(
            function,
        )) = tool
        else {
            panic!("expected flat function tool, got {tool:?}");
        };
        expect_that!(&function.name, eq("get_weather"));
        expect_that!(
            function.description.as_deref(),
            some(eq("Get the current weather"))
        );
        expect_that!(function.strict, some(eq(true)));
        expect_that!(
            function.parameters.as_ref(),
            some(eq(&json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            })))
        );

        let chat_params = params
            .into_chat_params()
            .expect("into_chat_params should succeed");
        let chat_tools = chat_params.tools.expect("chat params should carry tools");
        let [chat_tool] = chat_tools.as_slice() else {
            panic!("expected exactly one chat tool, got {chat_tools:?}");
        };
        let OpenAICompatibleTool::Function { function } = chat_tool else {
            panic!("expected chat function tool, got {chat_tool:?}");
        };
        expect_that!(&function.name, eq("get_weather"));
        expect_that!(function.strict, eq(true));
    }

    #[gtest]
    fn test_nested_chat_tool_shape_still_accepted() {
        let body = json!({
            "model": "gpt-5",
            "input": "What's the weather in Paris?",
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object"}
                }
            }]
        });
        let params: OpenAICompatibleResponsesParams =
            serde_json::from_value(body).expect("nested chat tool body should deserialize");
        let tools = params.tools.as_deref().expect("tools should be present");
        let [tool] = tools else {
            panic!("expected exactly one tool, got {tools:?}");
        };
        let OpenAICompatibleResponsesTool::Chat(OpenAICompatibleTool::Function { function }) = tool
        else {
            panic!("expected nested chat function tool, got {tool:?}");
        };
        expect_that!(&function.name, eq("get_weather"));
    }

    #[gtest]
    fn test_flat_responses_custom_tool_deserializes() {
        let tool: OpenAICompatibleResponsesTool = serde_json::from_value(json!({
            "type": "custom",
            "name": "code_exec",
            "description": "Executes code"
        }))
        .expect("flat custom tool should deserialize");
        let OpenAICompatibleResponsesTool::Flat(OpenAICompatibleResponsesFlatTool::Custom(custom)) =
            &tool
        else {
            panic!("expected flat custom tool, got {tool:?}");
        };
        expect_that!(&custom.name, eq("code_exec"));

        let chat_tool: OpenAICompatibleTool = tool.into();
        let OpenAICompatibleTool::Custom { custom } = &chat_tool else {
            panic!("expected chat custom tool, got {chat_tool:?}");
        };
        expect_that!(&custom.name, eq("code_exec"));
    }

    #[gtest]
    fn test_flat_function_tool_without_parameters_or_strict() {
        // `parameters` and `strict` are optional in the Responses API.
        let tool: OpenAICompatibleResponsesTool = serde_json::from_value(json!({
            "type": "function",
            "name": "noop"
        }))
        .expect("flat tool without parameters/strict should deserialize");
        let chat_tool: OpenAICompatibleTool = tool.into();
        let OpenAICompatibleTool::Function { function } = &chat_tool else {
            panic!("expected chat function tool, got {chat_tool:?}");
        };
        expect_that!(&function.name, eq("noop"));
        expect_that!(function.strict, eq(false));
    }

    #[gtest]
    fn test_text_format_json_schema_maps_to_nested_response_format() {
        let body = json!({
            "model": "gpt-5",
            "input": "Give me a city",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "city",
                    "description": "A city",
                    "schema": {"type": "object", "properties": {"name": {"type": "string"}}},
                    "strict": true
                },
                "verbosity": "low"
            }
        });
        let params: OpenAICompatibleResponsesParams =
            serde_json::from_value(body).expect("body with text.format should deserialize");
        // Declared fields must not leak into `unknown_fields` (regression: they
        // used to trigger "Ignoring unknown fields" warnings and get dropped).
        expect_that!(params.unknown_fields.contains_key("text"), eq(false));

        let chat_params = params
            .into_chat_params()
            .expect("into_chat_params should succeed");
        let Some(OpenAICompatibleResponseFormat::JsonSchema { json_schema }) =
            chat_params.response_format
        else {
            panic!(
                "expected nested json_schema response format, got {:?}",
                chat_params.response_format
            );
        };
        expect_that!(&json_schema.name, eq("city"));
        expect_that!(json_schema.description.as_deref(), some(eq("A city")));
        expect_that!(
            json_schema.schema.as_ref(),
            some(eq(
                &json!({"type": "object", "properties": {"name": {"type": "string"}}})
            ))
        );
        expect_that!(json_schema.strict, eq(true));
        expect_that!(chat_params.verbosity.as_deref(), some(eq("low")));
    }

    #[gtest]
    fn test_text_format_text_and_json_object_shapes() {
        for (format, expected) in [
            (
                json!({"type": "text"}),
                OpenAICompatibleResponseFormat::Text,
            ),
            (
                json!({"type": "json_object"}),
                OpenAICompatibleResponseFormat::JsonObject,
            ),
        ] {
            let body = json!({
                "model": "gpt-5",
                "input": "hi",
                "text": {"format": format}
            });
            let params: OpenAICompatibleResponsesParams =
                serde_json::from_value(body).expect("body should deserialize");
            let chat_params = params
                .into_chat_params()
                .expect("into_chat_params should succeed");
            expect_that!(chat_params.response_format, some(eq(&expected)));
        }
    }

    #[gtest]
    fn test_json_schema_format_defaults_strict_and_optional_fields() {
        let body = json!({
            "model": "gpt-5",
            "input": "hi",
            "text": {"format": {"type": "json_schema", "name": "bare"}}
        });
        let params: OpenAICompatibleResponsesParams =
            serde_json::from_value(body).expect("body should deserialize");
        let chat_params = params
            .into_chat_params()
            .expect("into_chat_params should succeed");
        let Some(OpenAICompatibleResponseFormat::JsonSchema { json_schema }) =
            chat_params.response_format
        else {
            panic!("expected json_schema response format");
        };
        expect_that!(&json_schema.name, eq("bare"));
        expect_that!(json_schema.description, none());
        expect_that!(json_schema.schema, none());
        expect_that!(json_schema.strict, eq(false));
    }

    #[gtest]
    fn test_reasoning_effort_and_service_tier_passthrough() {
        let body = json!({
            "model": "gpt-5",
            "input": "hi",
            "reasoning": {"effort": "high", "summary": "auto"},
            "service_tier": "priority"
        });
        let params: OpenAICompatibleResponsesParams =
            serde_json::from_value(body).expect("body should deserialize");
        expect_that!(params.unknown_fields.contains_key("reasoning"), eq(false));
        expect_that!(
            params.unknown_fields.contains_key("service_tier"),
            eq(false)
        );

        let chat_params = params
            .into_chat_params()
            .expect("into_chat_params should succeed");
        expect_that!(chat_params.reasoning_effort.as_deref(), some(eq("high")));
        expect_that!(chat_params.service_tier, some(eq(&ServiceTier::Priority)));
    }

    #[gtest]
    fn test_responses_specific_fields_still_land_in_unknown_fields() {
        // Fields we deliberately don't map keep flowing into `unknown_fields`
        // so the "Ignoring unknown fields" warning keeps its early-warning role.
        let body = json!({
            "model": "gpt-5",
            "input": "hi",
            "background": true,
            "previous_response_id": "resp_123"
        });
        let params: OpenAICompatibleResponsesParams =
            serde_json::from_value(body).expect("body should deserialize");
        expect_that!(params.unknown_fields.contains_key("background"), eq(true));
        expect_that!(
            params.unknown_fields.contains_key("previous_response_id"),
            eq(true)
        );
    }
}

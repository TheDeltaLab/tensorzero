// Modified by Delta-AI under Apache 2.0
//! OpenAI Completions API (`POST /v1/completions`) types.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use uuid::Uuid;

use crate::cache::CacheParamsOptions;
use crate::config::Namespace;
use crate::endpoints::inference::{InferenceCredentials, InferenceParams, InferenceResponse};
use crate::endpoints::openai_compatible::types::chat_completions::{
    OpenAICompatibleFinishReason, OpenAICompatibleMessage, OpenAICompatibleParams,
    OpenAICompatibleStreamOptions, OpenAICompatibleUserMessage, deserialize_stop_sequences,
    process_chat_content,
};
use crate::endpoints::openai_compatible::types::usage::OpenAICompatibleUsage;
use crate::error::Error;
use crate::inference::types::{FinishReason, current_timestamp};

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum CompletionPrompt {
    String(String),
    Strings(Vec<String>),
}

#[derive(Clone, Debug, Deserialize)]
pub struct OpenAICompatibleCompletionParams {
    pub model: String,
    pub prompt: CompletionPrompt,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub stream: Option<bool>,
    pub stream_options: Option<OpenAICompatibleStreamOptions>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub top_p: Option<f32>,
    pub seed: Option<u32>,
    pub n: Option<u32>,
    #[serde(default, deserialize_with = "deserialize_stop_sequences")]
    pub stop: Option<Vec<String>>,
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

impl OpenAICompatibleCompletionParams {
    pub fn into_chat_params(self) -> Result<OpenAICompatibleParams, Error> {
        let prompt = match self.prompt {
            CompletionPrompt::String(text) => text,
            CompletionPrompt::Strings(parts) => parts.join("\n"),
        };
        Ok(OpenAICompatibleParams {
            messages: vec![OpenAICompatibleMessage::User(OpenAICompatibleUserMessage {
                content: Value::String(prompt),
            })],
            model: self.model,
            frequency_penalty: self.frequency_penalty,
            max_tokens: self.max_tokens,
            presence_penalty: self.presence_penalty,
            seed: self.seed,
            stream: self.stream,
            stream_options: self.stream_options,
            temperature: self.temperature,
            top_p: self.top_p,
            n: self.n,
            stop: self.stop,
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

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OpenAICompatibleCompletionChoice {
    pub text: String,
    pub index: u32,
    pub logprobs: Option<()>,
    pub finish_reason: OpenAICompatibleFinishReason,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OpenAICompatibleCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u32,
    pub model: String,
    pub choices: Vec<OpenAICompatibleCompletionChoice>,
    pub usage: OpenAICompatibleUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub episode_id: Option<String>,
}

impl From<(InferenceResponse, String)> for OpenAICompatibleCompletionResponse {
    fn from((inference_response, response_model_prefix): (InferenceResponse, String)) -> Self {
        match inference_response {
            InferenceResponse::Chat(response) => {
                let (content, _tool_calls, _extra) = process_chat_content(response.content);
                OpenAICompatibleCompletionResponse {
                    id: response.inference_id.to_string(),
                    object: "text_completion".to_string(),
                    created: current_timestamp() as u32,
                    model: format!("{response_model_prefix}{}", response.variant_name),
                    choices: vec![OpenAICompatibleCompletionChoice {
                        text: content.unwrap_or_default(),
                        index: 0,
                        logprobs: None,
                        finish_reason: response.finish_reason.unwrap_or(FinishReason::Stop).into(),
                    }],
                    usage: response.usage.into(),
                    episode_id: Some(response.episode_id.to_string()),
                }
            }
            InferenceResponse::Json(response) => OpenAICompatibleCompletionResponse {
                id: response.inference_id.to_string(),
                object: "text_completion".to_string(),
                created: current_timestamp() as u32,
                model: format!("{response_model_prefix}{}", response.variant_name),
                choices: vec![OpenAICompatibleCompletionChoice {
                    text: response.output.raw.unwrap_or_default(),
                    index: 0,
                    logprobs: None,
                    finish_reason: response.finish_reason.unwrap_or(FinishReason::Stop).into(),
                }],
                usage: response.usage.into(),
                episode_id: Some(response.episode_id.to_string()),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_completion_prompt_string_converts_to_user_message() {
        let params: OpenAICompatibleCompletionParams = serde_json::from_value(json!({
            "model": "dummy::good",
            "prompt": "Hello",
            "max_tokens": 16,
        }))
        .unwrap();
        let chat = params.into_chat_params().unwrap();
        assert_eq!(chat.model, "dummy::good");
        assert_eq!(chat.max_tokens, Some(16));
        assert_eq!(chat.messages.len(), 1);
    }

    #[test]
    fn test_completion_prompt_array_joins() {
        let params: OpenAICompatibleCompletionParams = serde_json::from_value(json!({
            "model": "dummy::good",
            "prompt": ["Hello", "world"],
        }))
        .unwrap();
        let chat = params.into_chat_params().unwrap();
        match &chat.messages[0] {
            OpenAICompatibleMessage::User(msg) => {
                assert_eq!(msg.content, Value::String("Hello\nworld".to_string()));
            }
            _ => panic!("expected user message"),
        }
    }
}

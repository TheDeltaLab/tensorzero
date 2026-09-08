// Modified by Delta-AI under Apache 2.0
//! Provider configuration types, loading, and dispatch.
//!
//! Extracted from `tensorzero-core`'s `model.rs` so that provider-related
//! changes don't recompile all of core (Delta-AI fork).

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::borrow::Cow;
use url::Url;

use strum::VariantNames;
use tensorzero_derive::TensorZeroDeserialize;
use tensorzero_error::{Error, ErrorDetails};
use tensorzero_http::TensorzeroHttpClient;
use tensorzero_inference_types::credentials::{
    CredentialLocation, CredentialLocationOrHardcoded, CredentialLocationWithFallback,
    EndpointLocation, ModelProviderRequestInfo, ProviderInferenceRequest,
};
use tensorzero_inference_types::provider_trait::{InferenceProvider, WrappedProvider};
use tensorzero_inference_types::utils::get_mock_provider_api_base;
use tensorzero_inference_types::{
    BatchRequestRow, ModelInferenceRequest, PeekableProviderInferenceResponseStream,
    PollBatchInferenceResponse, ProviderInferenceResponse, StartBatchProviderInferenceResponse,
};
use tensorzero_stored_config::{
    StoredContentBlockType, StoredCredentialLocation, StoredCredentialLocationOrHardcoded,
    StoredCredentialLocationWithFallback, StoredEndpointLocation, StoredHostedProviderKind,
    StoredOpenAIAPIType, StoredProviderConfig,
};
use tensorzero_types::ApiType;
use tensorzero_types::inference_params::InferenceCredentials;

use crate::default_credentials::{
    AnthropicKind, AzureKind, DeepSeekKind, FireworksKind, GoogleAIStudioGeminiKind, GroqKind,
    HyperbolicKind, MistralKind, OpenAIKind, OpenRouterKind, ProviderKind,
    ProviderTypeDefaultCredentials, SGLangKind, TGIKind, TogetherKind, VLLMKind, XAIKind,
};
use crate::provider_types::ProviderTypesConfig;
use crate::providers::aws_bedrock::build_aws_bedrock_provider_config;
use crate::providers::aws_sagemaker::{AWSSagemakerProvider, build_aws_sagemaker_config};
#[cfg(any(test, feature = "e2e_tests"))]
use crate::providers::dummy::DummyProvider;
use crate::providers::google_ai_studio_gemini::GoogleAIStudioGeminiProvider;
use crate::providers::hyperbolic::HyperbolicProvider;
use crate::providers::openai::{ContentBlockType, OpenAIAPIType};
use crate::providers::sglang::SGLangProvider;
use crate::providers::tgi::TGIProvider;
use crate::providers::{
    anthropic::AnthropicProvider, aws_bedrock::AWSBedrockProvider, azure::AzureProvider,
    deepseek::DeepSeekProvider, fireworks::FireworksProvider,
    gcp_vertex_anthropic::GCPVertexAnthropicProvider, gcp_vertex_gemini::GCPVertexGeminiProvider,
    groq::GroqProvider, mistral::MistralProvider, openai::OpenAIProvider,
    openrouter::OpenRouterProvider, together::TogetherProvider, vllm::VLLMProvider,
    xai::XAIProvider,
};

impl From<StoredHostedProviderKind> for HostedProviderKind {
    fn from(stored: StoredHostedProviderKind) -> Self {
        match stored {
            StoredHostedProviderKind::OpenAI => Self::OpenAI,
            StoredHostedProviderKind::TGI => Self::TGI,
        }
    }
}

impl From<StoredOpenAIAPIType> for OpenAIAPIType {
    fn from(stored: StoredOpenAIAPIType) -> Self {
        match stored {
            StoredOpenAIAPIType::ChatCompletions => Self::ChatCompletions,
            StoredOpenAIAPIType::Responses => Self::Responses,
        }
    }
}

impl From<StoredContentBlockType> for ContentBlockType {
    fn from(stored: StoredContentBlockType) -> Self {
        match stored {
            StoredContentBlockType::ImageUrl => Self::ImageUrl,
            StoredContentBlockType::File => Self::File,
            StoredContentBlockType::InputAudio => Self::InputAudio,
        }
    }
}

impl From<&ContentBlockType> for StoredContentBlockType {
    fn from(value: &ContentBlockType) -> Self {
        match value {
            ContentBlockType::ImageUrl => StoredContentBlockType::ImageUrl,
            ContentBlockType::File => StoredContentBlockType::File,
            ContentBlockType::InputAudio => StoredContentBlockType::InputAudio,
        }
    }
}

fn parse_optional_url(url: Option<String>, field_name: &str) -> Result<Option<Url>, Error> {
    url.map(|u| {
        u.parse::<Url>().map_err(|e| {
            Error::new(ErrorDetails::Config {
                message: format!("Failed to parse `{field_name}` URL: {e}"),
            })
        })
    })
    .transpose()
}

fn parse_url(url: String, field_name: &str) -> Result<Url, Error> {
    url.parse::<Url>().map_err(|e| {
        Error::new(ErrorDetails::Config {
            message: format!("Failed to parse `{field_name}` URL: {e}"),
        })
    })
}

impl TryFrom<StoredProviderConfig> for UninitializedProviderConfig {
    type Error = Error;

    fn try_from(stored: StoredProviderConfig) -> Result<Self, Error> {
        match stored {
            StoredProviderConfig::Anthropic {
                model_name,
                api_base,
                api_key_location,
                beta_structured_outputs,
                provider_tools,
            } => Ok(Self::Anthropic {
                model_name,
                api_base: parse_optional_url(api_base, "api_base")?,
                api_key_location: api_key_location.map(CredentialLocationWithFallback::from),
                beta_structured_outputs,
                provider_tools: provider_tools.unwrap_or_default(),
            }),
            StoredProviderConfig::AWSBedrock {
                model_id,
                region,
                allow_auto_detect_region,
                endpoint_url,
                api_key,
                access_key_id,
                secret_access_key,
                session_token,
            } => Ok(Self::AWSBedrock {
                model_id,
                region: region.map(CredentialLocationOrHardcoded::from),
                allow_auto_detect_region: allow_auto_detect_region.unwrap_or_default(),
                endpoint_url: endpoint_url.map(CredentialLocationOrHardcoded::from),
                api_key: api_key.map(CredentialLocation::from),
                access_key_id: access_key_id.map(CredentialLocation::from),
                secret_access_key: secret_access_key.map(CredentialLocation::from),
                session_token: session_token.map(CredentialLocation::from),
            }),
            StoredProviderConfig::AWSSagemaker {
                endpoint_name,
                model_name,
                region,
                allow_auto_detect_region,
                hosted_provider,
                endpoint_url,
                access_key_id,
                secret_access_key,
                session_token,
            } => Ok(Self::AWSSagemaker {
                endpoint_name,
                model_name,
                region: region.map(CredentialLocationOrHardcoded::from),
                allow_auto_detect_region: allow_auto_detect_region.unwrap_or_default(),
                hosted_provider: hosted_provider.into(),
                endpoint_url: endpoint_url.map(CredentialLocationOrHardcoded::from),
                access_key_id: access_key_id.map(CredentialLocation::from),
                secret_access_key: secret_access_key.map(CredentialLocation::from),
                session_token: session_token.map(CredentialLocation::from),
            }),
            StoredProviderConfig::Azure {
                deployment_id,
                endpoint,
                api_key_location,
            } => Ok(Self::Azure {
                deployment_id,
                endpoint: EndpointLocation::from(endpoint),
                api_key_location: api_key_location.map(CredentialLocationWithFallback::from),
            }),
            StoredProviderConfig::GCPVertexAnthropic {
                model_id,
                location,
                project_id,
                credential_location,
                provider_tools,
            } => Ok(Self::GCPVertexAnthropic {
                model_id,
                location,
                project_id,
                credential_location: credential_location.map(CredentialLocationWithFallback::from),
                provider_tools: provider_tools.unwrap_or_default(),
            }),
            StoredProviderConfig::GCPVertexGemini {
                model_id,
                endpoint_id,
                location,
                project_id,
                credential_location,
            } => Ok(Self::GCPVertexGemini {
                model_id,
                endpoint_id,
                location,
                project_id,
                credential_location: credential_location.map(CredentialLocationWithFallback::from),
            }),
            StoredProviderConfig::GoogleAIStudioGemini {
                model_name,
                api_key_location,
            } => Ok(Self::GoogleAIStudioGemini {
                model_name,
                api_key_location: api_key_location.map(CredentialLocationWithFallback::from),
            }),
            StoredProviderConfig::Groq {
                model_name,
                api_key_location,
                reasoning_format,
            } => Ok(Self::Groq {
                model_name,
                api_key_location: api_key_location.map(CredentialLocationWithFallback::from),
                reasoning_format,
            }),
            StoredProviderConfig::Hyperbolic {
                model_name,
                api_key_location,
            } => Ok(Self::Hyperbolic {
                model_name,
                api_key_location: api_key_location.map(CredentialLocationWithFallback::from),
            }),
            StoredProviderConfig::Fireworks {
                model_name,
                api_key_location,
                parse_think_blocks,
            } => Ok(Self::Fireworks {
                model_name,
                api_key_location: api_key_location.map(CredentialLocationWithFallback::from),
                parse_think_blocks,
            }),
            StoredProviderConfig::Mistral {
                model_name,
                api_key_location,
                prompt_mode,
            } => Ok(Self::Mistral {
                model_name,
                api_key_location: api_key_location.map(CredentialLocationWithFallback::from),
                prompt_mode,
            }),
            StoredProviderConfig::OpenAI {
                model_name,
                api_base,
                api_key_location,
                api_type,
                include_encrypted_reasoning,
                provider_tools,
                content_type_overrides,
            } => Ok(Self::OpenAI {
                model_name,
                api_base: parse_optional_url(api_base, "api_base")?,
                api_key_location: api_key_location.map(CredentialLocationWithFallback::from),
                api_type: api_type.map(OpenAIAPIType::from).unwrap_or_default(),
                include_encrypted_reasoning: include_encrypted_reasoning.unwrap_or_default(),
                provider_tools: provider_tools.unwrap_or_default(),
                content_type_overrides: content_type_overrides
                    .unwrap_or_default()
                    .into_iter()
                    .map(|(k, v)| (k, ContentBlockType::from(v)))
                    .collect(),
            }),
            StoredProviderConfig::OpenRouter {
                model_name,
                api_key_location,
            } => Ok(Self::OpenRouter {
                model_name,
                api_key_location: api_key_location.map(CredentialLocationWithFallback::from),
            }),
            StoredProviderConfig::Together {
                model_name,
                api_key_location,
                parse_think_blocks,
            } => Ok(Self::Together {
                model_name,
                api_key_location: api_key_location.map(CredentialLocationWithFallback::from),
                parse_think_blocks,
            }),
            StoredProviderConfig::VLLM {
                model_name,
                api_base,
                api_key_location,
            } => Ok(Self::VLLM {
                model_name,
                api_base: parse_url(api_base, "api_base")?,
                api_key_location: api_key_location.map(CredentialLocationWithFallback::from),
            }),
            StoredProviderConfig::XAI {
                model_name,
                api_key_location,
            } => Ok(Self::XAI {
                model_name,
                api_key_location: api_key_location.map(CredentialLocationWithFallback::from),
            }),
            StoredProviderConfig::TGI {
                api_base,
                api_key_location,
            } => Ok(Self::TGI {
                api_base: parse_url(api_base, "api_base")?,
                api_key_location: api_key_location.map(CredentialLocationWithFallback::from),
            }),
            StoredProviderConfig::SGLang {
                model_name,
                api_base,
                api_key_location,
            } => Ok(Self::SGLang {
                model_name,
                api_base: parse_url(api_base, "api_base")?,
                api_key_location: api_key_location.map(CredentialLocationWithFallback::from),
            }),
            StoredProviderConfig::DeepSeek {
                model_name,
                api_key_location,
            } => Ok(Self::DeepSeek {
                model_name,
                api_key_location: api_key_location.map(CredentialLocationWithFallback::from),
            }),
            #[cfg(any(test, feature = "e2e_tests"))]
            StoredProviderConfig::Dummy {
                model_name,
                api_key_location,
            } => Ok(Self::Dummy {
                model_name,
                api_key_location: api_key_location
                    .map(serde_json::from_value)
                    .transpose()
                    .map_err(|e| {
                        Error::new(ErrorDetails::Config {
                            message: format!("invalid Dummy api_key_location: {e}"),
                        })
                    })?,
            }),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
#[derive(ts_rs::TS)]
#[ts(export)]
pub enum ProviderConfig {
    Anthropic(AnthropicProvider),
    #[serde(rename = "aws_bedrock")]
    AWSBedrock(AWSBedrockProvider),
    #[serde(rename = "aws_sagemaker")]
    AWSSagemaker(AWSSagemakerProvider),
    Azure(AzureProvider),
    DeepSeek(DeepSeekProvider),
    Fireworks(FireworksProvider),
    #[serde(rename = "gcp_vertex_anthropic")]
    GCPVertexAnthropic(GCPVertexAnthropicProvider),
    #[serde(rename = "gcp_vertex_gemini")]
    GCPVertexGemini(GCPVertexGeminiProvider),
    #[serde(rename = "google_ai_studio_gemini")]
    GoogleAIStudioGemini(GoogleAIStudioGeminiProvider),
    Groq(GroqProvider),
    Hyperbolic(HyperbolicProvider),
    Mistral(MistralProvider),
    OpenAI(OpenAIProvider),
    OpenRouter(OpenRouterProvider),
    #[serde(rename = "sglang")]
    SGLang(SGLangProvider),
    #[serde(rename = "tgi")]
    TGI(TGIProvider),
    Together(TogetherProvider),
    #[serde(rename = "vllm")]
    VLLM(VLLMProvider),
    #[serde(rename = "xai")]
    XAI(XAIProvider),
    #[cfg(any(test, feature = "e2e_tests"))]
    Dummy(DummyProvider),
}

impl ProviderConfig {
    pub fn thought_block_provider_type(&self) -> Cow<'static, str> {
        match self {
            ProviderConfig::Anthropic(_) => {
                Cow::Borrowed(crate::providers::anthropic::PROVIDER_TYPE)
            }
            ProviderConfig::AWSBedrock(_) => {
                Cow::Borrowed(crate::providers::aws_bedrock::PROVIDER_TYPE)
            }
            // Note - none of our current  wrapped provider types emit thought blocks
            // If any of them ever start producing thoughts, we'll need to make sure that the `provider_type`
            // field uses `thought_block_provider_type` on the parent SageMaker provider.
            ProviderConfig::AWSSagemaker(sagemaker) => Cow::Owned(format!(
                "aws_sagemaker::{}",
                sagemaker
                    .hosted_provider
                    .thought_block_provider_type_suffix()
            )),
            ProviderConfig::Azure(_) => Cow::Borrowed(crate::providers::azure::PROVIDER_TYPE),
            ProviderConfig::DeepSeek(_) => Cow::Borrowed(crate::providers::deepseek::PROVIDER_TYPE),
            ProviderConfig::Fireworks(_) => {
                Cow::Borrowed(crate::providers::fireworks::PROVIDER_TYPE)
            }
            ProviderConfig::GCPVertexAnthropic(_) => {
                Cow::Borrowed(crate::providers::gcp_vertex_anthropic::PROVIDER_TYPE)
            }
            ProviderConfig::GCPVertexGemini(_) => {
                Cow::Borrowed(crate::providers::gcp_vertex_gemini::PROVIDER_TYPE)
            }
            ProviderConfig::GoogleAIStudioGemini(_) => {
                Cow::Borrowed(crate::providers::google_ai_studio_gemini::PROVIDER_TYPE)
            }
            ProviderConfig::Groq(_) => Cow::Borrowed(crate::providers::groq::PROVIDER_TYPE),
            ProviderConfig::Hyperbolic(_) => {
                Cow::Borrowed(crate::providers::hyperbolic::PROVIDER_TYPE)
            }
            ProviderConfig::Mistral(_) => Cow::Borrowed(crate::providers::mistral::PROVIDER_TYPE),
            ProviderConfig::OpenAI(_) => Cow::Borrowed(crate::providers::openai::PROVIDER_TYPE),
            ProviderConfig::OpenRouter(_) => {
                Cow::Borrowed(crate::providers::openrouter::PROVIDER_TYPE)
            }
            ProviderConfig::SGLang(_) => Cow::Borrowed(crate::providers::sglang::PROVIDER_TYPE),
            ProviderConfig::TGI(_) => Cow::Borrowed(crate::providers::tgi::PROVIDER_TYPE),
            ProviderConfig::Together(_) => Cow::Borrowed(crate::providers::together::PROVIDER_TYPE),
            ProviderConfig::VLLM(_) => Cow::Borrowed(crate::providers::vllm::PROVIDER_TYPE),
            ProviderConfig::XAI(_) => Cow::Borrowed(crate::providers::xai::PROVIDER_TYPE),
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfig::Dummy(_) => Cow::Borrowed(crate::providers::dummy::PROVIDER_TYPE),
        }
    }

    /// Returns whether this provider supports provider tools (e.g., web_search, bash).
    /// This is an exhaustive match to ensure compile-time safety when adding new providers.
    pub fn supports_provider_tools(&self) -> bool {
        match self {
            // Providers that support provider tools
            ProviderConfig::Anthropic(_) => true,
            ProviderConfig::GCPVertexAnthropic(_) => true,
            ProviderConfig::OpenAI(provider) => provider.supports_provider_tools(),
            // Providers that do NOT support provider tools
            //
            // NB: AWS Bedrock supports them, but it's tricky because there are different fields for different models.
            //
            // Claude: uses `additionalModelRequestFields`
            //
            // ```
            // aws bedrock-runtime converse --model-id us.anthropic.claude-sonnet-4-5-20250929-v1:0 --messages '[{"role": "user", "content": [{"text": "Can you ping google.com using curl?"}]}]' --inference-config '{"maxTokens": 512}' --additional-model-request-fields '{"tools": [{"type": "bash_20250124", "name": "bash"}]}' --region us-east-1
            // ```
            //
            // Nova: uses `toolConfig` instead (https://docs.aws.amazon.com/nova/latest/nova2-userguide/web-grounding.html)
            ProviderConfig::AWSBedrock(_) => false,
            ProviderConfig::AWSSagemaker(_) => false,
            ProviderConfig::Azure(_) => false,
            ProviderConfig::DeepSeek(_) => false,
            ProviderConfig::Fireworks(_) => false,
            ProviderConfig::GCPVertexGemini(_) => false,
            ProviderConfig::GoogleAIStudioGemini(_) => false,
            ProviderConfig::Groq(_) => false,
            ProviderConfig::Hyperbolic(_) => false,
            ProviderConfig::Mistral(_) => false,
            ProviderConfig::OpenRouter(_) => false,
            ProviderConfig::SGLang(_) => false,
            ProviderConfig::TGI(_) => false,
            ProviderConfig::Together(_) => false,
            ProviderConfig::VLLM(_) => false,
            ProviderConfig::XAI(_) => false,
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfig::Dummy(_) => false,
        }
    }
}

/// Contains all providers which implement `SelfHostedProvider` - these providers
/// can be used as the target provider hosted by AWS Sagemaker
#[derive(ts_rs::TS, Clone, Debug, Deserialize, PartialEq, Serialize)]
#[ts(export)]
#[serde(rename_all = "lowercase")]
#[serde(deny_unknown_fields)]
pub enum HostedProviderKind {
    OpenAI,
    TGI,
}

#[derive(ts_rs::TS)]
#[ts(export, optional_fields)]
#[derive(Clone, Debug, PartialEq, TensorZeroDeserialize, VariantNames, Serialize)]
#[strum(serialize_all = "lowercase")]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
#[serde(deny_unknown_fields)]
pub enum UninitializedProviderConfig {
    Anthropic {
        model_name: String,
        #[ts(type = "string | null")]
        api_base: Option<Url>,
        #[ts(type = "string | null")]
        api_key_location: Option<CredentialLocationWithFallback>,
        #[serde(default)]
        beta_structured_outputs: Option<bool>,
        #[serde(default)]
        provider_tools: Vec<Value>,
    },
    #[strum(serialize = "aws_bedrock")]
    #[serde(rename = "aws_bedrock")]
    AWSBedrock {
        model_id: String,
        #[ts(type = "string | null")]
        region: Option<CredentialLocationOrHardcoded>,
        /// Deprecated: Use `region = "sdk"` instead to enable auto-detection.
        #[serde(default)]
        allow_auto_detect_region: bool,
        #[ts(type = "string | null")]
        endpoint_url: Option<CredentialLocationOrHardcoded>,
        /// API key for bearer token authentication (alternative to IAM credentials).
        /// If set, uses `Authorization: Bearer <token>` instead of SigV4 signing.
        #[ts(type = "string | null")]
        api_key: Option<CredentialLocation>,
        #[ts(type = "string | null")]
        access_key_id: Option<CredentialLocation>,
        #[ts(type = "string | null")]
        secret_access_key: Option<CredentialLocation>,
        #[ts(type = "string | null")]
        session_token: Option<CredentialLocation>,
    },
    #[strum(serialize = "aws_sagemaker")]
    #[serde(rename = "aws_sagemaker")]
    AWSSagemaker {
        endpoint_name: String,
        model_name: String,
        #[ts(type = "string | null")]
        region: Option<CredentialLocationOrHardcoded>,
        /// Deprecated: Use `region = "sdk"` instead to enable auto-detection.
        #[serde(default)]
        allow_auto_detect_region: bool,
        hosted_provider: HostedProviderKind,
        #[ts(type = "string | null")]
        endpoint_url: Option<CredentialLocationOrHardcoded>,
        #[ts(type = "string | null")]
        access_key_id: Option<CredentialLocation>,
        #[ts(type = "string | null")]
        secret_access_key: Option<CredentialLocation>,
        #[ts(type = "string | null")]
        session_token: Option<CredentialLocation>,
    },
    Azure {
        deployment_id: String,
        endpoint: EndpointLocation,
        #[ts(type = "string | null")]
        api_key_location: Option<CredentialLocationWithFallback>,
    },
    #[strum(serialize = "gcp_vertex_anthropic")]
    #[serde(rename = "gcp_vertex_anthropic")]
    GCPVertexAnthropic {
        model_id: String,
        location: String,
        project_id: String,
        #[ts(type = "string | null")]
        credential_location: Option<CredentialLocationWithFallback>,
        #[serde(default)]
        provider_tools: Vec<Value>,
    },
    #[strum(serialize = "gcp_vertex_gemini")]
    #[serde(rename = "gcp_vertex_gemini")]
    GCPVertexGemini {
        model_id: Option<String>,
        endpoint_id: Option<String>,
        location: String,
        project_id: String,
        #[ts(type = "string | null")]
        credential_location: Option<CredentialLocationWithFallback>,
    },
    #[strum(serialize = "google_ai_studio_gemini")]
    #[serde(rename = "google_ai_studio_gemini")]
    GoogleAIStudioGemini {
        model_name: String,
        #[ts(type = "string | null")]
        api_key_location: Option<CredentialLocationWithFallback>,
    },
    #[strum(serialize = "groq")]
    #[serde(rename = "groq")]
    Groq {
        model_name: String,
        #[ts(type = "string | null")]
        api_key_location: Option<CredentialLocationWithFallback>,
        reasoning_format: Option<String>,
    },
    Hyperbolic {
        model_name: String,
        #[ts(type = "string | null")]
        api_key_location: Option<CredentialLocationWithFallback>,
    },
    #[strum(serialize = "fireworks")]
    #[serde(rename = "fireworks")]
    Fireworks {
        model_name: String,
        #[ts(type = "string | null")]
        api_key_location: Option<CredentialLocationWithFallback>,
        #[serde(default)]
        parse_think_blocks: Option<bool>,
    },
    Mistral {
        model_name: String,
        #[ts(type = "string | null")]
        api_key_location: Option<CredentialLocationWithFallback>,
        prompt_mode: Option<String>,
    },
    OpenAI {
        model_name: String,
        api_base: Option<Url>,
        #[ts(type = "string | null")]
        api_key_location: Option<CredentialLocationWithFallback>,
        #[serde(default)]
        api_type: OpenAIAPIType,
        #[serde(default)]
        include_encrypted_reasoning: bool,
        #[serde(default)]
        provider_tools: Vec<Value>,
        #[serde(default)]
        content_type_overrides:
            std::collections::HashMap<String, crate::providers::openai::ContentBlockType>,
    },
    OpenRouter {
        model_name: String,
        #[ts(type = "string | null")]
        api_key_location: Option<CredentialLocationWithFallback>,
    },
    Together {
        model_name: String,
        #[ts(type = "string | null")]
        api_key_location: Option<CredentialLocationWithFallback>,
        #[serde(default)]
        parse_think_blocks: Option<bool>,
    },
    VLLM {
        model_name: String,
        api_base: Url,
        #[ts(type = "string | null")]
        api_key_location: Option<CredentialLocationWithFallback>,
    },
    XAI {
        model_name: String,
        #[ts(type = "string | null")]
        api_key_location: Option<CredentialLocationWithFallback>,
    },
    TGI {
        api_base: Url,
        #[ts(type = "string | null")]
        api_key_location: Option<CredentialLocationWithFallback>,
    },
    SGLang {
        model_name: String,
        api_base: Url,
        #[ts(type = "string | null")]
        api_key_location: Option<CredentialLocationWithFallback>,
    },
    DeepSeek {
        model_name: String,
        #[ts(type = "string | null")]
        api_key_location: Option<CredentialLocationWithFallback>,
    },
    #[cfg(any(test, feature = "e2e_tests"))]
    Dummy {
        model_name: String,
        #[ts(type = "string | null")]
        api_key_location: Option<CredentialLocationWithFallback>,
    },
}

impl From<&UninitializedProviderConfig> for StoredProviderConfig {
    fn from(config: &UninitializedProviderConfig) -> Self {
        match config {
            UninitializedProviderConfig::Anthropic {
                model_name,
                api_base,
                api_key_location,
                beta_structured_outputs,
                provider_tools,
            } => StoredProviderConfig::Anthropic {
                model_name: model_name.clone(),
                api_base: api_base.as_ref().map(ToString::to_string),
                api_key_location: api_key_location
                    .as_ref()
                    .map(StoredCredentialLocationWithFallback::from),
                beta_structured_outputs: *beta_structured_outputs,
                provider_tools: (!provider_tools.is_empty()).then(|| provider_tools.clone()),
            },
            UninitializedProviderConfig::AWSBedrock {
                model_id,
                region,
                allow_auto_detect_region,
                endpoint_url,
                api_key,
                access_key_id,
                secret_access_key,
                session_token,
            } => StoredProviderConfig::AWSBedrock {
                model_id: model_id.clone(),
                region: region
                    .as_ref()
                    .map(StoredCredentialLocationOrHardcoded::from),
                allow_auto_detect_region: Some(*allow_auto_detect_region),
                endpoint_url: endpoint_url
                    .as_ref()
                    .map(StoredCredentialLocationOrHardcoded::from),
                api_key: api_key.as_ref().map(StoredCredentialLocation::from),
                access_key_id: access_key_id.as_ref().map(StoredCredentialLocation::from),
                secret_access_key: secret_access_key
                    .as_ref()
                    .map(StoredCredentialLocation::from),
                session_token: session_token.as_ref().map(StoredCredentialLocation::from),
            },
            UninitializedProviderConfig::AWSSagemaker {
                endpoint_name,
                model_name,
                region,
                allow_auto_detect_region,
                hosted_provider,
                endpoint_url,
                access_key_id,
                secret_access_key,
                session_token,
            } => StoredProviderConfig::AWSSagemaker {
                endpoint_name: endpoint_name.clone(),
                model_name: model_name.clone(),
                region: region
                    .as_ref()
                    .map(StoredCredentialLocationOrHardcoded::from),
                allow_auto_detect_region: Some(*allow_auto_detect_region),
                hosted_provider: hosted_provider.clone().into(),
                endpoint_url: endpoint_url
                    .as_ref()
                    .map(StoredCredentialLocationOrHardcoded::from),
                access_key_id: access_key_id.as_ref().map(StoredCredentialLocation::from),
                secret_access_key: secret_access_key
                    .as_ref()
                    .map(StoredCredentialLocation::from),
                session_token: session_token.as_ref().map(StoredCredentialLocation::from),
            },
            UninitializedProviderConfig::Azure {
                deployment_id,
                endpoint,
                api_key_location,
            } => StoredProviderConfig::Azure {
                deployment_id: deployment_id.clone(),
                endpoint: StoredEndpointLocation::from(endpoint),
                api_key_location: api_key_location
                    .as_ref()
                    .map(StoredCredentialLocationWithFallback::from),
            },
            UninitializedProviderConfig::GCPVertexAnthropic {
                model_id,
                location,
                project_id,
                credential_location,
                provider_tools,
            } => StoredProviderConfig::GCPVertexAnthropic {
                model_id: model_id.clone(),
                location: location.clone(),
                project_id: project_id.clone(),
                credential_location: credential_location
                    .as_ref()
                    .map(StoredCredentialLocationWithFallback::from),
                provider_tools: (!provider_tools.is_empty()).then(|| provider_tools.clone()),
            },
            UninitializedProviderConfig::GCPVertexGemini {
                model_id,
                endpoint_id,
                location,
                project_id,
                credential_location,
            } => StoredProviderConfig::GCPVertexGemini {
                model_id: model_id.clone(),
                endpoint_id: endpoint_id.clone(),
                location: location.clone(),
                project_id: project_id.clone(),
                credential_location: credential_location
                    .as_ref()
                    .map(StoredCredentialLocationWithFallback::from),
            },
            UninitializedProviderConfig::GoogleAIStudioGemini {
                model_name,
                api_key_location,
            } => StoredProviderConfig::GoogleAIStudioGemini {
                model_name: model_name.clone(),
                api_key_location: api_key_location
                    .as_ref()
                    .map(StoredCredentialLocationWithFallback::from),
            },
            UninitializedProviderConfig::Groq {
                model_name,
                api_key_location,
                reasoning_format,
            } => StoredProviderConfig::Groq {
                model_name: model_name.clone(),
                api_key_location: api_key_location
                    .as_ref()
                    .map(StoredCredentialLocationWithFallback::from),
                reasoning_format: reasoning_format.clone(),
            },
            UninitializedProviderConfig::Hyperbolic {
                model_name,
                api_key_location,
            } => StoredProviderConfig::Hyperbolic {
                model_name: model_name.clone(),
                api_key_location: api_key_location
                    .as_ref()
                    .map(StoredCredentialLocationWithFallback::from),
            },
            UninitializedProviderConfig::Fireworks {
                model_name,
                api_key_location,
                parse_think_blocks,
            } => StoredProviderConfig::Fireworks {
                model_name: model_name.clone(),
                api_key_location: api_key_location
                    .as_ref()
                    .map(StoredCredentialLocationWithFallback::from),
                parse_think_blocks: *parse_think_blocks,
            },
            UninitializedProviderConfig::Mistral {
                model_name,
                api_key_location,
                prompt_mode,
            } => StoredProviderConfig::Mistral {
                model_name: model_name.clone(),
                api_key_location: api_key_location
                    .as_ref()
                    .map(StoredCredentialLocationWithFallback::from),
                prompt_mode: prompt_mode.clone(),
            },
            UninitializedProviderConfig::OpenAI {
                model_name,
                api_base,
                api_key_location,
                api_type,
                include_encrypted_reasoning,
                provider_tools,
                content_type_overrides,
            } => StoredProviderConfig::OpenAI {
                model_name: model_name.clone(),
                api_base: api_base.as_ref().map(ToString::to_string),
                api_key_location: api_key_location
                    .as_ref()
                    .map(StoredCredentialLocationWithFallback::from),
                api_type: Some((*api_type).into()),
                include_encrypted_reasoning: Some(*include_encrypted_reasoning),
                provider_tools: (!provider_tools.is_empty()).then(|| provider_tools.clone()),
                content_type_overrides: (!content_type_overrides.is_empty()).then(|| {
                    content_type_overrides
                        .iter()
                        .map(|(k, v)| (k.clone(), v.into()))
                        .collect()
                }),
            },
            UninitializedProviderConfig::OpenRouter {
                model_name,
                api_key_location,
            } => StoredProviderConfig::OpenRouter {
                model_name: model_name.clone(),
                api_key_location: api_key_location
                    .as_ref()
                    .map(StoredCredentialLocationWithFallback::from),
            },
            UninitializedProviderConfig::Together {
                model_name,
                api_key_location,
                parse_think_blocks,
            } => StoredProviderConfig::Together {
                model_name: model_name.clone(),
                api_key_location: api_key_location
                    .as_ref()
                    .map(StoredCredentialLocationWithFallback::from),
                parse_think_blocks: *parse_think_blocks,
            },
            UninitializedProviderConfig::VLLM {
                model_name,
                api_base,
                api_key_location,
            } => StoredProviderConfig::VLLM {
                model_name: model_name.clone(),
                api_base: api_base.to_string(),
                api_key_location: api_key_location
                    .as_ref()
                    .map(StoredCredentialLocationWithFallback::from),
            },
            UninitializedProviderConfig::XAI {
                model_name,
                api_key_location,
            } => StoredProviderConfig::XAI {
                model_name: model_name.clone(),
                api_key_location: api_key_location
                    .as_ref()
                    .map(StoredCredentialLocationWithFallback::from),
            },
            UninitializedProviderConfig::TGI {
                api_base,
                api_key_location,
            } => StoredProviderConfig::TGI {
                api_base: api_base.to_string(),
                api_key_location: api_key_location
                    .as_ref()
                    .map(StoredCredentialLocationWithFallback::from),
            },
            UninitializedProviderConfig::SGLang {
                model_name,
                api_base,
                api_key_location,
            } => StoredProviderConfig::SGLang {
                model_name: model_name.clone(),
                api_base: api_base.to_string(),
                api_key_location: api_key_location
                    .as_ref()
                    .map(StoredCredentialLocationWithFallback::from),
            },
            UninitializedProviderConfig::DeepSeek {
                model_name,
                api_key_location,
            } => StoredProviderConfig::DeepSeek {
                model_name: model_name.clone(),
                api_key_location: api_key_location
                    .as_ref()
                    .map(StoredCredentialLocationWithFallback::from),
            },
            #[cfg(any(test, feature = "e2e_tests"))]
            UninitializedProviderConfig::Dummy {
                model_name,
                api_key_location,
            } => StoredProviderConfig::Dummy {
                model_name: model_name.clone(),
                api_key_location: api_key_location
                    .as_ref()
                    .and_then(|value| serde_json::to_value(value).ok()),
            },
        }
    }
}

impl UninitializedProviderConfig {
    pub async fn load(
        self,
        provider_types: &ProviderTypesConfig,
        provider_type_default_credentials: &ProviderTypeDefaultCredentials,
        is_config_snapshot: bool,
    ) -> Result<ProviderConfig, Error> {
        Ok(match self {
            UninitializedProviderConfig::Anthropic {
                model_name,
                api_base,
                api_key_location,
                beta_structured_outputs,
                provider_tools,
            } => {
                // Only log a deprecation warning if this is a fresh config (not a snapshot)
                // since snapshot configs cannot be updated
                if !is_config_snapshot && beta_structured_outputs.is_some() {
                    tensorzero_inference_types::utils::deprecation_warning(
                        "The 'beta_structured_outputs' field is no longer necessary for `anthropic` providers",
                    );
                }
                ProviderConfig::Anthropic(AnthropicProvider::new(
                    model_name,
                    api_base,
                    AnthropicKind
                        .get_defaulted_credential(
                            api_key_location.as_ref(),
                            provider_type_default_credentials,
                        )
                        .await?,
                    provider_tools,
                ))
            }
            UninitializedProviderConfig::AWSBedrock {
                model_id,
                region,
                allow_auto_detect_region,
                endpoint_url,
                api_key,
                access_key_id,
                secret_access_key,
                session_token,
            } => {
                let (region, endpoint_url, auth) = build_aws_bedrock_provider_config(
                    region,
                    allow_auto_detect_region,
                    endpoint_url,
                    api_key,
                    access_key_id,
                    secret_access_key,
                    session_token,
                )
                .await?;

                ProviderConfig::AWSBedrock(AWSBedrockProvider::new(
                    model_id,
                    region,
                    endpoint_url,
                    auth,
                ))
            }
            UninitializedProviderConfig::AWSSagemaker {
                endpoint_name,
                region,
                allow_auto_detect_region,
                model_name,
                hosted_provider,
                endpoint_url,
                access_key_id,
                secret_access_key,
                session_token,
            } => {
                let aws_config = build_aws_sagemaker_config(
                    region,
                    allow_auto_detect_region,
                    endpoint_url,
                    access_key_id,
                    secret_access_key,
                    session_token,
                )?;

                let self_hosted: Box<dyn WrappedProvider + Send + Sync + 'static> =
                    match hosted_provider {
                        HostedProviderKind::OpenAI => Box::new(OpenAIProvider::new(
                            model_name,
                            None,
                            OpenAIKind
                                .get_defaulted_credential(
                                    Some(&CredentialLocationWithFallback::Single(CredentialLocation::None)),
                                    provider_type_default_credentials,
                                )
                                .await?,
                            // TODO - decide how to expose the responses api for wrapped providers
                            OpenAIAPIType::ChatCompletions,
                            false,
                            Vec::new(),
                            std::collections::HashMap::new(),
                            )?),
                        HostedProviderKind::TGI => Box::new(TGIProvider::new(
                            Url::parse("http://tensorzero-unreachable-domain-please-file-a-bug-report.invalid").map_err(|e| {
                                Error::new(ErrorDetails::InternalError { message: format!("Failed to parse fake TGI endpoint: `{e}`. This should never happen. Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new") })
                            })?,
                            TGIKind
                                .get_defaulted_credential(
                                    Some(&CredentialLocationWithFallback::Single(CredentialLocation::None)),
                                    provider_type_default_credentials,
                                )
                                .await?,
                        )),
                    };

                ProviderConfig::AWSSagemaker(
                    AWSSagemakerProvider::new(
                        endpoint_name,
                        self_hosted,
                        aws_config.region,
                        aws_config.endpoint_url,
                        aws_config.credentials,
                    )
                    .await?,
                )
            }
            UninitializedProviderConfig::Azure {
                deployment_id,
                endpoint: azure_endpoint,
                api_key_location,
            } => ProviderConfig::Azure(AzureProvider::new(
                deployment_id,
                azure_endpoint,
                AzureKind
                    .get_defaulted_credential(
                        api_key_location.as_ref(),
                        provider_type_default_credentials,
                    )
                    .await?,
            )?),
            UninitializedProviderConfig::Fireworks {
                model_name,
                api_key_location,
                parse_think_blocks,
            } => {
                if !is_config_snapshot && parse_think_blocks.is_some() {
                    tensorzero_inference_types::utils::deprecation_warning(
                        "The `parse_think_blocks` option for `fireworks` providers is deprecated and will be removed in a future release. Think blocks are now always parsed.",
                    );
                }
                ProviderConfig::Fireworks(FireworksProvider::new(
                    model_name,
                    FireworksKind
                        .get_defaulted_credential(
                            api_key_location.as_ref(),
                            provider_type_default_credentials,
                        )
                        .await?,
                ))
            }
            UninitializedProviderConfig::GCPVertexAnthropic {
                model_id,
                location,
                project_id,
                credential_location: api_key_location,
                provider_tools,
            } => {
                let credentials = crate::default_credentials::GCPVertexAnthropicKind
                    .get_defaulted_credential(
                        api_key_location.as_ref(),
                        provider_type_default_credentials,
                    )
                    .await?;
                ProviderConfig::GCPVertexAnthropic(GCPVertexAnthropicProvider::new(
                    model_id,
                    location,
                    project_id,
                    credentials,
                    provider_tools,
                ))
            }
            UninitializedProviderConfig::GCPVertexGemini {
                model_id,
                endpoint_id,
                location,
                project_id,
                credential_location: api_key_location,
            } => {
                let credentials = crate::default_credentials::GCPVertexGeminiKind
                    .get_defaulted_credential(
                        api_key_location.as_ref(),
                        provider_type_default_credentials,
                    )
                    .await?;
                let batch_config = match &provider_types.gcp_vertex_gemini {
                    Some(crate::provider_types::GCPVertexGeminiProviderTypeConfig {
                        batch:
                            Some(crate::provider_types::GCPBatchConfigType::CloudStorage(
                                crate::provider_types::GCPBatchConfigCloudStorage {
                                    input_uri_prefix,
                                    output_uri_prefix,
                                },
                            )),
                        ..
                    }) => Some(crate::providers::gcp_vertex_gemini::BatchConfig::new(
                        &project_id,
                        &location,
                        input_uri_prefix.clone(),
                        output_uri_prefix.clone(),
                    )),
                    _ => None,
                };
                let provider = GCPVertexGeminiProvider::new(
                    model_id,
                    endpoint_id,
                    location,
                    project_id,
                    credentials,
                    batch_config,
                )?;

                ProviderConfig::GCPVertexGemini(provider)
            }
            UninitializedProviderConfig::GoogleAIStudioGemini {
                model_name,
                api_key_location,
            } => ProviderConfig::GoogleAIStudioGemini(GoogleAIStudioGeminiProvider::new(
                model_name,
                GoogleAIStudioGeminiKind
                    .get_defaulted_credential(
                        api_key_location.as_ref(),
                        provider_type_default_credentials,
                    )
                    .await?,
            )?),
            UninitializedProviderConfig::Groq {
                model_name,
                api_key_location,
                reasoning_format,
            } => ProviderConfig::Groq(GroqProvider::new(
                model_name,
                GroqKind
                    .get_defaulted_credential(
                        api_key_location.as_ref(),
                        provider_type_default_credentials,
                    )
                    .await?,
                reasoning_format,
            )),
            UninitializedProviderConfig::Hyperbolic {
                model_name,
                api_key_location,
            } => ProviderConfig::Hyperbolic(HyperbolicProvider::new(
                model_name,
                HyperbolicKind
                    .get_defaulted_credential(
                        api_key_location.as_ref(),
                        provider_type_default_credentials,
                    )
                    .await?,
            )),
            UninitializedProviderConfig::Mistral {
                model_name,
                api_key_location,
                prompt_mode,
            } => ProviderConfig::Mistral(MistralProvider::new(
                model_name,
                MistralKind
                    .get_defaulted_credential(
                        api_key_location.as_ref(),
                        provider_type_default_credentials,
                    )
                    .await?,
                prompt_mode,
            )),
            UninitializedProviderConfig::OpenAI {
                model_name,
                api_base,
                api_key_location,
                api_type,
                include_encrypted_reasoning,
                provider_tools,
                content_type_overrides,
            } => {
                // Use mock API base for testing if set, otherwise defer to the API base set
                let api_base = get_mock_provider_api_base("openai").or(api_base);

                ProviderConfig::OpenAI(OpenAIProvider::new(
                    model_name,
                    api_base,
                    OpenAIKind
                        .get_defaulted_credential(
                            api_key_location.as_ref(),
                            provider_type_default_credentials,
                        )
                        .await?,
                    api_type,
                    include_encrypted_reasoning,
                    provider_tools,
                    content_type_overrides,
                )?)
            }
            UninitializedProviderConfig::OpenRouter {
                model_name,
                api_key_location,
            } => ProviderConfig::OpenRouter(OpenRouterProvider::new(
                model_name,
                OpenRouterKind
                    .get_defaulted_credential(
                        api_key_location.as_ref(),
                        provider_type_default_credentials,
                    )
                    .await?,
            )),
            UninitializedProviderConfig::Together {
                model_name,
                api_key_location,
                parse_think_blocks,
            } => {
                if !is_config_snapshot && parse_think_blocks.is_some() {
                    // Deprecation: #6502 - 2026.5+
                    tensorzero_inference_types::utils::deprecation_warning(
                        "The `parse_think_blocks` option for `together` providers is deprecated and will be removed in a future release. Think blocks are now always parsed.",
                    );
                }
                ProviderConfig::Together(TogetherProvider::new(
                    model_name,
                    TogetherKind
                        .get_defaulted_credential(
                            api_key_location.as_ref(),
                            provider_type_default_credentials,
                        )
                        .await?,
                ))
            }
            UninitializedProviderConfig::VLLM {
                model_name,
                api_base,
                api_key_location,
            } => ProviderConfig::VLLM(VLLMProvider::new(
                model_name,
                api_base,
                VLLMKind
                    .get_defaulted_credential(
                        api_key_location.as_ref(),
                        provider_type_default_credentials,
                    )
                    .await?,
            )),
            UninitializedProviderConfig::XAI {
                model_name,
                api_key_location,
            } => ProviderConfig::XAI(XAIProvider::new(
                model_name,
                XAIKind
                    .get_defaulted_credential(
                        api_key_location.as_ref(),
                        provider_type_default_credentials,
                    )
                    .await?,
            )),
            UninitializedProviderConfig::SGLang {
                model_name,
                api_base,
                api_key_location,
            } => ProviderConfig::SGLang(SGLangProvider::new(
                model_name,
                api_base,
                SGLangKind
                    .get_defaulted_credential(
                        api_key_location.as_ref(),
                        provider_type_default_credentials,
                    )
                    .await?,
            )),
            UninitializedProviderConfig::TGI {
                api_base,
                api_key_location,
            } => ProviderConfig::TGI(TGIProvider::new(
                api_base,
                TGIKind
                    .get_defaulted_credential(
                        api_key_location.as_ref(),
                        provider_type_default_credentials,
                    )
                    .await?,
            )),
            UninitializedProviderConfig::DeepSeek {
                model_name,
                api_key_location,
            } => ProviderConfig::DeepSeek(DeepSeekProvider::new(
                model_name,
                DeepSeekKind
                    .get_defaulted_credential(
                        api_key_location.as_ref(),
                        provider_type_default_credentials,
                    )
                    .await?,
            )),
            #[cfg(any(test, feature = "e2e_tests"))]
            UninitializedProviderConfig::Dummy {
                model_name,
                api_key_location,
            } => ProviderConfig::Dummy(DummyProvider::new(model_name, api_key_location)?),
        })
    }
}

impl From<HostedProviderKind> for StoredHostedProviderKind {
    fn from(kind: HostedProviderKind) -> Self {
        match kind {
            HostedProviderKind::OpenAI => StoredHostedProviderKind::OpenAI,
            HostedProviderKind::TGI => StoredHostedProviderKind::TGI,
        }
    }
}

impl From<OpenAIAPIType> for StoredOpenAIAPIType {
    fn from(api_type: OpenAIAPIType) -> Self {
        match api_type {
            OpenAIAPIType::ChatCompletions => StoredOpenAIAPIType::ChatCompletions,
            OpenAIAPIType::Responses => StoredOpenAIAPIType::Responses,
        }
    }
}

impl ProviderConfig {
    /// The provider type string (e.g., "openai", "anthropic")
    pub fn provider_type(&self) -> &'static str {
        match self {
            ProviderConfig::Anthropic(_) => crate::providers::anthropic::PROVIDER_TYPE,
            ProviderConfig::AWSBedrock(_) => crate::providers::aws_bedrock::PROVIDER_TYPE,
            ProviderConfig::AWSSagemaker(_) => crate::providers::aws_sagemaker::PROVIDER_TYPE,
            ProviderConfig::Azure(_) => crate::providers::azure::PROVIDER_TYPE,
            ProviderConfig::Fireworks(_) => crate::providers::fireworks::PROVIDER_TYPE,
            ProviderConfig::GCPVertexAnthropic(_) => {
                crate::providers::gcp_vertex_anthropic::PROVIDER_TYPE
            }
            ProviderConfig::GCPVertexGemini(_) => {
                crate::providers::gcp_vertex_gemini::PROVIDER_TYPE
            }
            ProviderConfig::GoogleAIStudioGemini(_) => {
                crate::providers::google_ai_studio_gemini::PROVIDER_TYPE
            }
            ProviderConfig::Groq(_) => crate::providers::groq::PROVIDER_TYPE,
            ProviderConfig::Hyperbolic(_) => crate::providers::hyperbolic::PROVIDER_TYPE,
            ProviderConfig::Mistral(_) => crate::providers::mistral::PROVIDER_TYPE,
            ProviderConfig::OpenAI(_) => crate::providers::openai::PROVIDER_TYPE,
            ProviderConfig::OpenRouter(_) => crate::providers::openrouter::PROVIDER_TYPE,
            ProviderConfig::Together(_) => crate::providers::together::PROVIDER_TYPE,
            ProviderConfig::VLLM(_) => crate::providers::vllm::PROVIDER_TYPE,
            ProviderConfig::XAI(_) => crate::providers::xai::PROVIDER_TYPE,
            ProviderConfig::TGI(_) => crate::providers::tgi::PROVIDER_TYPE,
            ProviderConfig::SGLang(_) => crate::providers::sglang::PROVIDER_TYPE,
            ProviderConfig::DeepSeek(_) => crate::providers::deepseek::PROVIDER_TYPE,
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfig::Dummy(_) => crate::providers::dummy::PROVIDER_TYPE,
        }
    }

    /// The API type used by this provider (e.g., ChatCompletions, Responses)
    pub fn api_type(&self) -> ApiType {
        match self {
            ProviderConfig::OpenAI(provider) => provider.api_type().into(),
            // All other providers use ChatCompletions API
            _ => ApiType::ChatCompletions,
        }
    }

    /// The model name, if the provider has a meaningful one.
    pub fn model_name(&self) -> Option<&str> {
        match self {
            ProviderConfig::Anthropic(provider) => Some(provider.model_name()),
            ProviderConfig::AWSBedrock(provider) => Some(provider.model_id()),
            // SageMaker doesn't have a meaningful model name concept, as we just invoke an endpoint
            ProviderConfig::AWSSagemaker(_) => None,
            ProviderConfig::Azure(provider) => Some(provider.deployment_id()),
            ProviderConfig::Fireworks(provider) => Some(provider.model_name()),
            ProviderConfig::GCPVertexAnthropic(provider) => Some(provider.model_id()),
            ProviderConfig::GCPVertexGemini(provider) => Some(provider.model_or_endpoint_id()),
            ProviderConfig::GoogleAIStudioGemini(provider) => Some(provider.model_name()),
            ProviderConfig::Groq(provider) => Some(provider.model_name()),
            ProviderConfig::Hyperbolic(provider) => Some(provider.model_name()),
            ProviderConfig::Mistral(provider) => Some(provider.model_name()),
            ProviderConfig::OpenAI(provider) => Some(provider.model_name()),
            ProviderConfig::OpenRouter(provider) => Some(provider.model_name()),
            ProviderConfig::Together(provider) => Some(provider.model_name()),
            ProviderConfig::VLLM(provider) => Some(provider.model_name()),
            ProviderConfig::XAI(provider) => Some(provider.model_name()),
            // TGI doesn't have a meaningful model name
            ProviderConfig::TGI(_) => None,
            ProviderConfig::SGLang(provider) => Some(provider.model_name()),
            ProviderConfig::DeepSeek(provider) => Some(provider.model_name()),
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfig::Dummy(provider) => Some(provider.model_name()),
        }
    }

    pub async fn infer(
        &self,
        provider_request: ProviderInferenceRequest<'_>,
        http_client: &TensorzeroHttpClient,
        dynamic_api_keys: &InferenceCredentials,
        model_provider_info: &ModelProviderRequestInfo,
    ) -> Result<ProviderInferenceResponse, Error> {
        match self {
            ProviderConfig::Anthropic(provider) => {
                provider
                    .infer(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::AWSBedrock(provider) => {
                provider
                    .infer(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::AWSSagemaker(provider) => {
                provider
                    .infer(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::Azure(provider) => {
                provider
                    .infer(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::Fireworks(provider) => {
                provider
                    .infer(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::GCPVertexAnthropic(provider) => {
                provider
                    .infer(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::GCPVertexGemini(provider) => {
                provider
                    .infer(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::Groq(provider) => {
                provider
                    .infer(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::GoogleAIStudioGemini(provider) => {
                provider
                    .infer(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::Hyperbolic(provider) => {
                provider
                    .infer(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::Mistral(provider) => {
                provider
                    .infer(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::OpenAI(provider) => {
                provider
                    .infer(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::OpenRouter(provider) => {
                provider
                    .infer(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::Together(provider) => {
                provider
                    .infer(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::SGLang(provider) => {
                provider
                    .infer(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::VLLM(provider) => {
                provider
                    .infer(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::XAI(provider) => {
                provider
                    .infer(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::TGI(provider) => {
                provider
                    .infer(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::DeepSeek(provider) => {
                provider
                    .infer(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfig::Dummy(provider) => {
                provider
                    .infer(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
        }
    }

    pub async fn infer_stream(
        &self,
        provider_request: ProviderInferenceRequest<'_>,
        http_client: &TensorzeroHttpClient,
        dynamic_api_keys: &InferenceCredentials,
        model_provider_info: &ModelProviderRequestInfo,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        match self {
            ProviderConfig::Anthropic(provider) => {
                provider
                    .infer_stream(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::AWSBedrock(provider) => {
                provider
                    .infer_stream(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::AWSSagemaker(provider) => {
                provider
                    .infer_stream(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::Azure(provider) => {
                provider
                    .infer_stream(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::Fireworks(provider) => {
                provider
                    .infer_stream(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::GCPVertexAnthropic(provider) => {
                provider
                    .infer_stream(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::GCPVertexGemini(provider) => {
                provider
                    .infer_stream(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::GoogleAIStudioGemini(provider) => {
                provider
                    .infer_stream(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::Groq(provider) => {
                provider
                    .infer_stream(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::Hyperbolic(provider) => {
                provider
                    .infer_stream(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::Mistral(provider) => {
                provider
                    .infer_stream(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::OpenAI(provider) => {
                provider
                    .infer_stream(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::OpenRouter(provider) => {
                provider
                    .infer_stream(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::Together(provider) => {
                provider
                    .infer_stream(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::SGLang(provider) => {
                provider
                    .infer_stream(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::XAI(provider) => {
                provider
                    .infer_stream(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::VLLM(provider) => {
                provider
                    .infer_stream(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::TGI(provider) => {
                provider
                    .infer_stream(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            ProviderConfig::DeepSeek(provider) => {
                provider
                    .infer_stream(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfig::Dummy(provider) => {
                provider
                    .infer_stream(
                        provider_request,
                        http_client,
                        dynamic_api_keys,
                        model_provider_info,
                    )
                    .await
            }
        }
    }

    pub async fn start_batch_inference<'a>(
        &self,
        requests: &'a [ModelInferenceRequest<'a>],
        client: &'a TensorzeroHttpClient,
        api_keys: &'a InferenceCredentials,
    ) -> Result<StartBatchProviderInferenceResponse, Error> {
        match self {
            ProviderConfig::Anthropic(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::AWSBedrock(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::AWSSagemaker(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::Azure(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::Fireworks(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::GCPVertexAnthropic(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::GCPVertexGemini(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::GoogleAIStudioGemini(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::Groq(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::Hyperbolic(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::Mistral(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::OpenAI(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::OpenRouter(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::Together(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::SGLang(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::VLLM(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::XAI(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::DeepSeek(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::TGI(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfig::Dummy(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
        }
    }

    pub async fn poll_batch_inference<'a>(
        &self,
        batch_request: &'a BatchRequestRow<'_>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<PollBatchInferenceResponse, Error> {
        match self {
            ProviderConfig::Anthropic(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::AWSBedrock(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::AWSSagemaker(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::Azure(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::Fireworks(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::GCPVertexAnthropic(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::GCPVertexGemini(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::GoogleAIStudioGemini(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::Groq(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::Hyperbolic(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::Mistral(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::OpenAI(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::OpenRouter(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::TGI(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::Together(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::SGLang(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::VLLM(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::XAI(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::DeepSeek(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfig::Dummy(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
        }
    }
}

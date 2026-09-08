// Modified by Delta-AI under Apache 2.0
#![expect(clippy::unwrap_used)]

//! Test fixtures for provider tests.
//!
//! Extracted from `tensorzero-core` when providers moved to this crate.
//! These fixtures no longer depend on core's `ToolCallConfig` machinery:
//! `ProviderToolCallConfig` (from `tensorzero-inference-types`) is constructed
//! directly, and a minimal `ToolCallConfig` mirror is provided for tests that
//! construct or inspect tool configs.

use std::sync::Arc;

use lazy_static::lazy_static;
use serde_json::json;
use tensorzero_inference_types::{
    AllowedTools, FunctionToolDef, OpenAICustomTool, ProviderTool, ProviderToolCallConfig,
};
use tensorzero_types::ToolChoice;

use crate::jsonschema_util::JSONSchema;

pub const IMPLICIT_TOOL_DESCRIPTION: &str =
    "The implicit tool for enforcing JSON schema compliance";
pub const IMPLICIT_TOOL_NAME: &str = "respond_with_json";

/// Mirror of core's `StaticToolConfig` for tests.
#[derive(Debug, PartialEq, Clone)]
pub struct StaticToolConfig {
    pub description: String,
    pub parameters: JSONSchema,
    pub name: String,
    pub key: String,
    pub strict: bool,
}

/// Mirror of core's `DynamicToolConfig` for tests.
#[derive(Debug, PartialEq, Clone)]
pub struct DynamicToolConfig {
    pub description: String,
    pub parameters: JSONSchema,
    pub name: String,
    pub strict: bool,
}

/// Mirror of core's `ImplicitToolConfig` for tests.
#[derive(Debug, PartialEq, Clone)]
pub struct ImplicitToolConfig {
    pub parameters: JSONSchema,
}

/// Mirror of core's `FunctionToolConfig` for tests.
#[derive(Debug, PartialEq, Clone)]
pub enum FunctionToolConfig {
    Static(Arc<StaticToolConfig>),
    Dynamic(DynamicToolConfig),
    Implicit(ImplicitToolConfig),
}

impl FunctionToolConfig {
    pub fn description(&self) -> &str {
        match self {
            FunctionToolConfig::Static(config) => &config.description,
            FunctionToolConfig::Dynamic(config) => &config.description,
            FunctionToolConfig::Implicit(_) => IMPLICIT_TOOL_DESCRIPTION,
        }
    }

    pub fn parameters(&self) -> &serde_json::Value {
        match self {
            FunctionToolConfig::Static(config) => &config.parameters.value,
            FunctionToolConfig::Dynamic(config) => &config.parameters.value,
            FunctionToolConfig::Implicit(config) => &config.parameters.value,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            FunctionToolConfig::Static(config) => &config.name,
            FunctionToolConfig::Dynamic(config) => &config.name,
            FunctionToolConfig::Implicit(_) => IMPLICIT_TOOL_NAME,
        }
    }

    pub fn key(&self) -> &str {
        match self {
            FunctionToolConfig::Static(config) => &config.key,
            FunctionToolConfig::Dynamic(config) => &config.name,
            FunctionToolConfig::Implicit(_) => IMPLICIT_TOOL_NAME,
        }
    }

    pub fn strict(&self) -> bool {
        match self {
            FunctionToolConfig::Static(config) => config.strict,
            FunctionToolConfig::Dynamic(config) => config.strict,
            FunctionToolConfig::Implicit(_) => false,
        }
    }
}

/// Mirror of core's `ToolCallConfig` for tests.
#[derive(Debug, Default, Clone)]
pub struct ToolCallConfig {
    pub static_tools_available: Vec<FunctionToolConfig>,
    pub dynamic_tools_available: Vec<FunctionToolConfig>,
    pub provider_tools: Vec<ProviderTool>,
    pub openai_custom_tools: Vec<OpenAICustomTool>,
    pub tool_choice: ToolChoice,
    pub parallel_tool_calls: Option<bool>,
    pub allowed_tools: AllowedTools,
}

impl ToolCallConfig {
    pub fn with_tools_available(
        static_tools_available: Vec<FunctionToolConfig>,
        dynamic_tools_available: Vec<FunctionToolConfig>,
    ) -> Self {
        Self {
            static_tools_available,
            dynamic_tools_available,
            ..Default::default()
        }
    }
}

impl From<&ToolCallConfig> for ProviderToolCallConfig {
    fn from(config: &ToolCallConfig) -> Self {
        let tools = config
            .static_tools_available
            .iter()
            .chain(config.dynamic_tools_available.iter())
            .map(|tc| FunctionToolDef {
                name: tc.name().to_string(),
                description: tc.description().to_string(),
                parameters: tc.parameters().clone(),
                strict: tc.strict(),
            })
            .collect();
        ProviderToolCallConfig {
            tools,
            provider_tools: config.provider_tools.clone(),
            openai_custom_tools: config.openai_custom_tools.clone(),
            tool_choice: config.tool_choice.clone(),
            parallel_tool_calls: config.parallel_tool_calls,
            allowed_tools: config.allowed_tools.clone(),
        }
    }
}

lazy_static! {
    /// These are useful for tests which don't need mutable tools.
    pub static ref WEATHER_TOOL_CONFIG_STATIC: Arc<StaticToolConfig> = Arc::new(StaticToolConfig {
        name: "get_temperature".to_string(),
        key: "get_temperature".to_string(),
        description: "Get the current temperature in a given location".to_string(),
        parameters: JSONSchema::from_value(json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        })).unwrap(),
        strict: false,
    });
    pub static ref WEATHER_TOOL: FunctionToolConfig = FunctionToolConfig::Static(WEATHER_TOOL_CONFIG_STATIC.clone());
    pub static ref WEATHER_TOOL_CHOICE: ToolChoice = ToolChoice::Specific("get_temperature".to_string());
    pub static ref WEATHER_TOOL_CONFIG: ToolCallConfig = ToolCallConfig {
        tool_choice: ToolChoice::Specific("get_temperature".to_string()),
        ..ToolCallConfig::with_tools_available(
            vec![FunctionToolConfig::Static(WEATHER_TOOL_CONFIG_STATIC.clone())],
            vec![],
        )
    };
    pub static ref QUERY_TOOL_CONFIG_STATIC: Arc<StaticToolConfig> = Arc::new(StaticToolConfig {
        name: "query_articles".to_string(),
        key: "query_articles".to_string(),
        description: "Query articles from Wikipedia".to_string(),
        parameters: JSONSchema::from_value(json!({
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "year": {"type": "integer"}
            },
            "required": ["query", "year"]
        })).unwrap(),
        strict: true,
    });
    pub static ref QUERY_TOOL: FunctionToolConfig = FunctionToolConfig::Static(QUERY_TOOL_CONFIG_STATIC.clone());
    pub static ref ANY_TOOL_CHOICE: ToolChoice = ToolChoice::Required;
    pub static ref MULTI_TOOL_CONFIG: ToolCallConfig = ToolCallConfig {
        tool_choice: ToolChoice::Required,
        parallel_tool_calls: Some(true),
        ..ToolCallConfig::with_tools_available(
            vec![
                FunctionToolConfig::Static(WEATHER_TOOL_CONFIG_STATIC.clone()),
                FunctionToolConfig::Static(QUERY_TOOL_CONFIG_STATIC.clone())
            ],
            vec![],
        )
    };
    pub static ref WEATHER_PROVIDER_TOOL_CONFIG: ProviderToolCallConfig =
        ProviderToolCallConfig::from(&*WEATHER_TOOL_CONFIG);
    pub static ref MULTI_PROVIDER_TOOL_CONFIG: ProviderToolCallConfig =
        ProviderToolCallConfig::from(&*MULTI_TOOL_CONFIG);
}

// For use in tests which need a mutable tool config.
pub fn get_temperature_tool_config() -> ToolCallConfig {
    let weather_tool = FunctionToolConfig::Static(WEATHER_TOOL_CONFIG_STATIC.clone());
    ToolCallConfig {
        tool_choice: ToolChoice::Specific("get_temperature".to_string()),
        parallel_tool_calls: Some(false),
        ..ToolCallConfig::with_tools_available(vec![weather_tool], vec![])
    }
}

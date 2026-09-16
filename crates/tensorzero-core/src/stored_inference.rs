// Modified by Delta-AI under Apache 2.0
use std::{collections::HashMap, sync::Arc};

use crate::client::InferenceParams;
use crate::config::Config;
use crate::error::{Error, ErrorDetails};
use crate::function::FunctionConfig;
use crate::inference::types::extra_body::DynamicExtraBody;
use crate::inference::types::extra_body::UnfilteredInferenceExtraBody;
#[cfg(feature = "pyo3")]
use crate::inference::types::pyo3_helpers::{
    content_block_chat_output_to_python, serialize_to_dict, uuid_to_python,
};
use crate::inference::types::stored_input::StoredInput;
use crate::inference::types::{
    ContentBlockChatOutput, FunctionType, JsonInferenceOutput, ModelInput, ResolvedInput, Text,
};
use crate::tool::{StaticToolConfig, ToolCallConfigDatabaseInsert, deserialize_optional_tool_info};
use crate::variant::{VariantConfig, chat_completion::prepare_model_input};
use chrono::{DateTime, Utc};
#[cfg(feature = "pyo3")]
use pyo3::types::PyList;
#[cfg(feature = "pyo3")]
use pyo3::{IntoPyObjectExt, prelude::*};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tensorzero_derive::TensorZeroDeserialize;
use tensorzero_inference_types::tool::DynamicToolParams;
use uuid::Uuid;

/// This trait is used to represent a stored sample of data.
/// It should contain all the methods used by `render_samples`
/// from the stored sample of data so that we can abstract over the
/// different places where we could get training samples from, notably
/// datasets and stored inferences.
pub trait StoredSample {
    fn function_name(&self) -> &str;
    fn into_input(self) -> Option<StoredInput>;
    fn input(&self) -> Option<&StoredInput>;
    fn input_mut(&mut self) -> Option<&mut StoredInput>;
    fn owned_simple_info(self) -> SimpleStoredSampleInfo;
}

/// Utility struct that contains the information needed for a RenderedSample
/// that is just copied over from the StoredSample.
pub struct SimpleStoredSampleInfo {
    pub function_name: String,
    pub function_type: FunctionType,
    pub input: Option<StoredInput>,
    pub episode_id: Option<Uuid>,
    pub inference_id: Option<Uuid>,
    pub output: Option<Vec<ContentBlockChatOutput>>,
    pub stored_output: Option<StoredOutput>,
    pub dispreferred_outputs: Vec<Vec<ContentBlockChatOutput>>,
    pub tool_params: Option<ToolCallConfigDatabaseInsert>,
    pub output_schema: Option<Value>,
    pub tags: HashMap<String, String>,
}

/// Wire variant of StoredInference for API responses with Python/TypeScript bindings
/// This one should be used in all public interfaces
#[derive(ts_rs::TS, Clone, Debug, JsonSchema, PartialEq, Serialize, TensorZeroDeserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum StoredInference {
    #[schemars(title = "StoredInferenceChat")]
    Chat(StoredChatInference),
    #[schemars(title = "StoredInferenceJson")]
    Json(StoredJsonInference),
}

impl std::fmt::Display for StoredInference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

impl StoredInference {
    pub fn id(&self) -> Uuid {
        match self {
            StoredInference::Json(inference) => inference.inference_id,
            StoredInference::Chat(inference) => inference.inference_id,
        }
    }
}

impl StoredInferenceDatabase {
    /// Convert to wire type, properly handling tool params by subtracting static tools
    pub fn into_stored_inference(self) -> Result<StoredInference, Error> {
        match self {
            StoredInferenceDatabase::Chat(chat) => {
                Ok(StoredInference::Chat(chat.into_stored_inference()))
            }
            StoredInferenceDatabase::Json(json) => Ok(StoredInference::Json(json)),
        }
    }
}

impl StoredInference {
    /// Convert to storage type, converting tool params from wire format to storage format
    pub fn to_storage(self, config: &Config) -> Result<StoredInferenceDatabase, Error> {
        match self {
            StoredInference::Chat(chat) => {
                let function_config = config.get_function(&chat.function_name)?;
                Ok(StoredInferenceDatabase::Chat(
                    chat.to_storage(&function_config, &config.tools)?,
                ))
            }
            StoredInference::Json(json) => Ok(StoredInferenceDatabase::Json(json)),
        }
    }
}

impl StoredChatInference {
    /// Convert to storage type, properly handling tool params with function config
    pub fn to_storage(
        self,
        function_config: &FunctionConfig,
        static_tools: &HashMap<String, Arc<StaticToolConfig>>,
    ) -> Result<StoredChatInferenceDatabase, Error> {
        let tool_params = function_config
            .dynamic_tool_params_to_database_insert(self.tool_params, static_tools)?
            .unwrap_or_default();

        Ok(StoredChatInferenceDatabase {
            function_name: self.function_name,
            variant_name: self.variant_name,
            input: self.input,
            output: self.output,
            dispreferred_outputs: self.dispreferred_outputs,
            timestamp: self.timestamp,
            episode_id: self.episode_id,
            inference_id: self.inference_id,
            tool_params: Some(tool_params),
            tags: self.tags,
            extra_body: self.extra_body,
            inference_params: self.inference_params,
            processing_time_ms: self.processing_time_ms,
            ttft_ms: self.ttft_ms,
            snapshot_hash: self.snapshot_hash,
            error: self.error,
        })
    }
}

/// Storage variant of StoredInference for database operations (no Python/TypeScript bindings)
#[derive(Clone, Debug, PartialEq, Serialize, TensorZeroDeserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum StoredInferenceDatabase {
    Chat(StoredChatInferenceDatabase),
    Json(StoredJsonInference),
}

impl std::fmt::Display for StoredInferenceDatabase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

impl StoredInferenceDatabase {
    pub fn id(&self) -> Uuid {
        match self {
            StoredInferenceDatabase::Json(inference) => inference.inference_id,
            StoredInferenceDatabase::Chat(inference) => inference.inference_id,
        }
    }

    pub fn timestamp(&self) -> DateTime<Utc> {
        match self {
            StoredInferenceDatabase::Json(inference) => inference.timestamp,
            StoredInferenceDatabase::Chat(inference) => inference.timestamp,
        }
    }
}

/// Wire variant of StoredChatInference for API responses with Python/TypeScript bindings
#[derive(ts_rs::TS, Clone, Debug, Deserialize, PartialEq, Serialize, JsonSchema)]
#[ts(export)]
pub struct StoredChatInference {
    pub function_name: String,
    pub variant_name: String,
    #[ts(optional)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<StoredInput>,
    #[ts(optional)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<Vec<ContentBlockChatOutput>>,
    #[serde(default)]
    pub dispreferred_outputs: Vec<Vec<ContentBlockChatOutput>>,
    #[schemars(with = "String")]
    pub timestamp: DateTime<Utc>,
    pub episode_id: Uuid,
    pub inference_id: Uuid,
    #[serde(flatten)]
    #[serde(default)]
    pub tool_params: DynamicToolParams,
    #[serde(default)]
    pub tags: HashMap<String, String>,
    #[serde(default)]
    #[ts(optional)]
    #[ts(as = "Option<Vec<DynamicExtraBody>>")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_body: Option<UnfilteredInferenceExtraBody>,
    #[ts(optional)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inference_params: Option<InferenceParams>,
    #[ts(optional)]
    pub processing_time_ms: Option<u64>,
    #[ts(optional)]
    pub ttft_ms: Option<u64>,
    #[ts(optional)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot_hash: Option<String>,
    /// Serialized error tree, present only on failed inference rows.
    #[ts(optional)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl std::fmt::Display for StoredChatInference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

impl StoredChatInferenceDatabase {
    /// Convert to wire type, converting tool params from storage format to wire format
    pub fn into_stored_inference(self) -> StoredChatInference {
        StoredChatInference {
            function_name: self.function_name,
            variant_name: self.variant_name,
            input: self.input,
            output: self.output,
            dispreferred_outputs: self.dispreferred_outputs,
            timestamp: self.timestamp,
            episode_id: self.episode_id,
            inference_id: self.inference_id,
            tool_params: self.tool_params.map(Into::into).unwrap_or_default(),
            tags: self.tags,
            extra_body: self.extra_body,
            inference_params: self.inference_params,
            processing_time_ms: self.processing_time_ms,
            ttft_ms: self.ttft_ms,
            snapshot_hash: self.snapshot_hash,
            error: self.error,
        }
    }
}

/// Storage variant of StoredChatInference for database operations (no Python/TypeScript bindings)
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct StoredChatInferenceDatabase {
    pub function_name: String,
    pub variant_name: String,
    #[serde(default)]
    pub input: Option<StoredInput>,
    #[serde(default)]
    pub output: Option<Vec<ContentBlockChatOutput>>,
    #[serde(default)]
    pub dispreferred_outputs: Vec<Vec<ContentBlockChatOutput>>,
    pub timestamp: DateTime<Utc>,
    pub episode_id: Uuid,
    pub inference_id: Uuid,
    #[serde(flatten, deserialize_with = "deserialize_optional_tool_info")]
    pub tool_params: Option<ToolCallConfigDatabaseInsert>,
    #[serde(default)]
    pub tags: HashMap<String, String>,
    #[serde(default)]
    pub extra_body: Option<UnfilteredInferenceExtraBody>,
    #[serde(default)]
    pub inference_params: Option<InferenceParams>,
    pub processing_time_ms: Option<u64>,
    pub ttft_ms: Option<u64>,
    #[serde(default)]
    pub snapshot_hash: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
}

impl std::fmt::Display for StoredChatInferenceDatabase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[derive(ts_rs::TS, Clone, Debug, Deserialize, PartialEq, Serialize, JsonSchema)]
#[ts(export)]
pub struct StoredJsonInference {
    pub function_name: String,
    pub variant_name: String,
    #[ts(optional)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<StoredInput>,
    #[ts(optional)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<JsonInferenceOutput>,
    #[serde(default)]
    pub dispreferred_outputs: Vec<JsonInferenceOutput>,
    #[schemars(with = "String")]
    pub timestamp: DateTime<Utc>,
    pub episode_id: Uuid,
    pub inference_id: Uuid,
    #[ts(optional)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<Value>,
    #[serde(default)]
    pub tags: HashMap<String, String>,
    #[serde(default)]
    #[ts(optional)]
    #[ts(as = "Option<Vec<DynamicExtraBody>>")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_body: Option<UnfilteredInferenceExtraBody>,
    #[serde(default)]
    #[ts(optional)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_params: Option<InferenceParams>,
    #[ts(optional)]
    pub processing_time_ms: Option<u64>,
    #[ts(optional)]
    pub ttft_ms: Option<u64>,
    #[ts(optional)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot_hash: Option<String>,
    /// Serialized error tree, present only on failed inference rows.
    #[ts(optional)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl std::fmt::Display for StoredJsonInference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

impl StoredSample for StoredInferenceDatabase {
    fn input_mut(&mut self) -> Option<&mut StoredInput> {
        match self {
            StoredInferenceDatabase::Chat(example) => example.input.as_mut(),
            StoredInferenceDatabase::Json(example) => example.input.as_mut(),
        }
    }

    fn input(&self) -> Option<&StoredInput> {
        match self {
            StoredInferenceDatabase::Chat(example) => example.input.as_ref(),
            StoredInferenceDatabase::Json(example) => example.input.as_ref(),
        }
    }

    fn into_input(self) -> Option<StoredInput> {
        match self {
            StoredInferenceDatabase::Chat(example) => example.input,
            StoredInferenceDatabase::Json(example) => example.input,
        }
    }

    fn function_name(&self) -> &str {
        match self {
            StoredInferenceDatabase::Chat(example) => &example.function_name,
            StoredInferenceDatabase::Json(example) => &example.function_name,
        }
    }

    fn owned_simple_info(self) -> SimpleStoredSampleInfo {
        match self {
            StoredInferenceDatabase::Chat(example) => SimpleStoredSampleInfo {
                function_name: example.function_name,
                function_type: FunctionType::Chat,
                input: example.input,
                episode_id: Some(example.episode_id),
                inference_id: Some(example.inference_id),
                output: example.output.clone(),
                stored_output: example.output.map(StoredOutput::Chat),
                dispreferred_outputs: example.dispreferred_outputs,
                tool_params: example.tool_params,
                output_schema: None,
                tags: example.tags,
            },
            StoredInferenceDatabase::Json(example) => {
                let output = example
                    .output
                    .as_ref()
                    .map(|o| json_output_to_content_block_chat_output(o.clone()));
                let stored_output = example.output.map(StoredOutput::Json);
                let dispreferred_outputs = example
                    .dispreferred_outputs
                    .into_iter()
                    .map(json_output_to_content_block_chat_output)
                    .collect();
                SimpleStoredSampleInfo {
                    function_name: example.function_name,
                    function_type: FunctionType::Json,
                    input: example.input,
                    episode_id: Some(example.episode_id),
                    inference_id: Some(example.inference_id),
                    output,
                    stored_output,
                    dispreferred_outputs,
                    tool_params: None,
                    output_schema: example.output_schema,
                    tags: example.tags,
                }
            }
        }
    }
}

fn json_output_to_content_block_chat_output(
    output: JsonInferenceOutput,
) -> Vec<ContentBlockChatOutput> {
    match output.raw {
        Some(raw) => vec![ContentBlockChatOutput::Text(Text { text: raw })],
        None => vec![],
    }
}

#[derive(ts_rs::TS, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[ts(export)]
#[serde(untagged)]
pub enum StoredOutput {
    Chat(Vec<ContentBlockChatOutput>),
    Json(JsonInferenceOutput),
}

/// Represents an inference that has been prepared for fine-tuning.
/// This is constructed by rendering a StoredInference with a variant for messages
/// and by resolving all network resources (e.g. images).
/// This is a wire type - it uses DynamicToolParams and has Python/TypeScript bindings.
#[cfg_attr(feature = "pyo3", pyclass(str))]
#[derive(ts_rs::TS, Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(any(feature = "e2e_tests", test), derive(PartialEq))]
#[ts(export)]
pub struct RenderedSample {
    pub function_name: String,
    pub function_type: FunctionType,
    pub input: ModelInput,
    pub stored_input: StoredInput,
    pub output: Option<Vec<ContentBlockChatOutput>>,
    pub stored_output: Option<StoredOutput>,
    pub dispreferred_outputs: Vec<Vec<ContentBlockChatOutput>>,
    pub episode_id: Option<Uuid>,
    pub inference_id: Option<Uuid>,
    pub tool_params: DynamicToolParams,
    pub output_schema: Option<Value>,
    pub tags: HashMap<String, String>,
}

impl RenderedSample {}

#[cfg(feature = "pyo3")]
impl RenderedSample {
    #[getter]
    pub fn get_input(&self) -> ModelInput {
        self.input.clone()
    }

    #[getter]
    pub fn get_output<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if let Some(output) = &self.output {
            let output = output
                .iter()
                .map(|x| content_block_chat_output_to_python(py, x.clone()))
                .collect::<PyResult<Vec<_>>>()?;
            PyList::new(py, output).map(Bound::into_any)
        } else {
            Ok(py.None().into_bound(py))
        }
    }

    #[getter]
    pub fn get_stored_output<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if let Some(stored_output) = &self.stored_output {
            match stored_output {
                StoredOutput::Chat(output) => {
                    let output = output
                        .iter()
                        .map(|x| content_block_chat_output_to_python(py, x.clone()))
                        .collect::<PyResult<Vec<_>>>()?;
                    PyList::new(py, output).map(Bound::into_any)
                }
                StoredOutput::Json(output) => Ok(output.clone().into_py_any(py)?.into_bound(py)),
            }
        } else {
            Ok(py.None().into_bound(py))
        }
    }

    #[getter]
    pub fn get_dispreferred_outputs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let dispreferred_outputs = self
            .dispreferred_outputs
            .iter()
            .map(|x| {
                x.iter()
                    .map(|y| content_block_chat_output_to_python(py, y.clone()))
                    .collect::<PyResult<Vec<_>>>()
            })
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, dispreferred_outputs).map(Bound::into_any)
    }

    #[getter]
    pub fn get_output_schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        serialize_to_dict(py, self.output_schema.clone()).map(|x| x.into_bound(py))
    }

    #[getter]
    pub fn get_allowed_tools(&self) -> Option<Vec<String>> {
        self.tool_params.allowed_tools.clone()
    }

    #[getter]
    pub fn get_additional_tools<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.tool_params
            .additional_tools
            .clone()
            .into_bound_py_any(py)
    }

    // Note: We're intentionally skipping tool_choice as it's not exposed in the Python API

    #[getter]
    pub fn get_parallel_tool_calls(&self) -> Option<bool> {
        self.tool_params.parallel_tool_calls
    }

    #[getter]
    pub fn get_provider_tools<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        serialize_to_dict(py, &self.tool_params.provider_tools).map(|x| x.into_bound(py))
    }

    #[getter]
    pub fn get_episode_id<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.episode_id {
            Some(id) => uuid_to_python(py, id),
            None => Ok(py.None().into_bound(py)),
        }
    }

    #[getter]
    pub fn get_inference_id<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.inference_id {
            Some(id) => uuid_to_python(py, id),
            None => Ok(py.None().into_bound(py)),
        }
    }

    pub fn __repr__(&self) -> String {
        self.to_string()
    }

    #[getter]
    pub fn get_tags(&self) -> HashMap<String, String> {
        self.tags.clone()
    }

    #[getter]
    pub fn get_stored_input(&self) -> StoredInput {
        self.stored_input.clone()
    }
}

impl std::fmt::Display for RenderedSample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Serialize the rendered inference to pretty-printed JSON
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

/// Convert a StoredInference's input to a ModelInput.
/// `variants` should be a map from function name to variant name, i.e. what variant to use for a particular function
/// as the stored inference is being rendered.
/// This does not handle resolving network resources (e.g. images).
async fn render_model_input(
    resolved_input: &ResolvedInput,
    function_name: &str,
    config: &Config,
    variants: &HashMap<String, String>,
) -> Result<ModelInput, Error> {
    let variant_name = variants.get(function_name).ok_or_else(|| {
        Error::new(ErrorDetails::MissingFunctionInVariants {
            function_name: function_name.to_string(),
        })
    })?;
    let function_config = config.get_function(function_name)?;
    let variant_config = function_config
        .variants()
        .get(variant_name)
        .ok_or_else(|| {
            Error::new(ErrorDetails::UnknownVariant {
                name: variant_name.clone(),
            })
        })?;
    let VariantConfig::ChatCompletion(chat_completion_config) = &variant_config.inner else {
        return Err(Error::new(ErrorDetails::InvalidVariantForOptimization {
            function_name: function_name.to_string(),
            variant_name: variant_name.clone(),
        }));
    };
    prepare_model_input(
        resolved_input.system.as_ref(),
        &resolved_input.messages,
        &config.templates,
        chat_completion_config.templates(),
    )
    .await
}

/// Render an impl StoredSample to a RenderedStoredInference.
/// `variants` should be a map from function name to variant name, i.e. what variant to use for a particular function
/// as the inference example is being rendered.
///
/// This does not handle resolving network resources (e.g. images).
pub async fn render_stored_sample<T: StoredSample>(
    stored_sample: T,
    resolved_input: ResolvedInput,
    config: &Config,
    variants: &HashMap<String, String>,
) -> Result<RenderedSample, Error> {
    let SimpleStoredSampleInfo {
        function_name,
        function_type,
        input: _,
        output,
        stored_output,
        dispreferred_outputs,
        tool_params,
        output_schema,
        episode_id,
        inference_id,
        tags,
    } = stored_sample.owned_simple_info();
    let model_input = render_model_input(&resolved_input, &function_name, config, variants).await?;

    // Convert tool_params from storage format to wire format
    let dynamic_tool_params = tool_params
        .map(|tp| tp.into())
        // should default for JSON functions or functions with no tools to a default DynamicToolParams
        // where everything is empty
        .unwrap_or_default();

    Ok(RenderedSample {
        function_name,
        function_type,
        episode_id,
        inference_id,
        input: model_input,
        stored_input: resolved_input.into_stored_input(),
        output,
        stored_output,
        dispreferred_outputs,
        tool_params: dynamic_tool_params,
        output_schema,
        tags,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, SchemaData};
    use crate::endpoints::inference::InferenceParams;
    use crate::function::{FunctionConfig, FunctionConfigChat, FunctionConfigJson};
    use crate::inference::types::System;
    use crate::inference::types::{ContentBlockChatOutput, JsonInferenceOutput, Text};
    use crate::jsonschema_util::JSONSchema;
    use crate::tool::{ToolCallConfig, ToolChoice};
    use std::sync::Arc;
    use tensorzero_inference_types::tool::DynamicToolParams;

    /// Helper to create a test config with the functions registered
    fn create_test_config() -> Config {
        let mut config = Config::default();

        // Add the test_function (Chat function)
        config.functions.insert(
            "test_function".to_string(),
            Arc::new(FunctionConfig::Chat(FunctionConfigChat {
                variants: Default::default(),
                schemas: SchemaData::default(),
                tools: vec![],
                tool_choice: ToolChoice::Auto,
                parallel_tool_calls: None,
                description: None,
                all_explicit_templates_names: Default::default(),
            })),
        );

        // Add the json_function (Json function)
        config.functions.insert(
            "json_function".to_string(),
            Arc::new(FunctionConfig::Json(FunctionConfigJson {
                variants: Default::default(),
                schemas: SchemaData::default(),
                output_schema: JSONSchema::default(),
                json_mode_tool_call_config: ToolCallConfig::default(),
                description: None,
                all_explicit_template_names: Default::default(),
            })),
        );

        config
    }

    /// Helper to create a test StoredChatInference with all fields populated
    fn create_test_chat_inference() -> StoredChatInference {
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();

        StoredChatInference {
            error: None,
            function_name: "test_function".to_string(),
            variant_name: "test_variant".to_string(),
            input: Some(StoredInput {
                system: Some(System::Text("Test system prompt".to_string())),
                messages: vec![],
            }),
            output: Some(vec![
                ContentBlockChatOutput::Text(Text {
                    text: "Test output 1".to_string(),
                }),
                ContentBlockChatOutput::Text(Text {
                    text: "Test output 2".to_string(),
                }),
            ]),
            dispreferred_outputs: vec![],
            timestamp: DateTime::parse_from_rfc3339("2024-01-01T00:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            episode_id,
            inference_id,
            tool_params: DynamicToolParams::default(),
            tags: {
                let mut tags = HashMap::new();
                tags.insert("key1".to_string(), "value1".to_string());
                tags.insert("key2".to_string(), "value2".to_string());
                tags
            },
            extra_body: Some(UnfilteredInferenceExtraBody::default()),
            inference_params: Some(InferenceParams::default()),
            processing_time_ms: None,
            ttft_ms: None,
            snapshot_hash: None,
        }
    }

    /// Helper to create a test StoredJsonInference with all fields populated
    fn create_test_json_inference() -> StoredJsonInference {
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();

        StoredJsonInference {
            error: None,
            function_name: "json_function".to_string(),
            variant_name: "json_variant".to_string(),
            input: Some(StoredInput {
                system: Some(System::Text("JSON system prompt".to_string())),
                messages: vec![],
            }),
            output: Some(JsonInferenceOutput {
                raw: Some(r#"{"result": "test"}"#.to_string()),
                parsed: Some(serde_json::json!({"result": "test"})),
            }),
            dispreferred_outputs: vec![],
            timestamp: DateTime::parse_from_rfc3339("2024-01-01T00:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            episode_id,
            inference_id,
            output_schema: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "result": {"type": "string"}
                }
            })),
            tags: {
                let mut tags = HashMap::new();
                tags.insert("json_key".to_string(), "json_value".to_string());
                tags
            },
            extra_body: Some(UnfilteredInferenceExtraBody::default()),
            inference_params: Some(InferenceParams::default()),
            processing_time_ms: None,
            ttft_ms: None,
            snapshot_hash: None,
        }
    }

    /// Helper to create a test RenderedSample for Chat function
    fn create_test_chat_rendered_sample() -> RenderedSample {
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();

        RenderedSample {
            function_name: "test_function".to_string(),
            function_type: FunctionType::Chat,
            input: ModelInput {
                system: Some("Test system prompt".to_string()),
                messages: vec![],
            },
            stored_input: StoredInput {
                system: Some(System::Text("Test system prompt".to_string())),
                messages: vec![],
            },
            output: Some(vec![
                ContentBlockChatOutput::Text(Text {
                    text: "Test output 1".to_string(),
                }),
                ContentBlockChatOutput::Text(Text {
                    text: "Test output 2".to_string(),
                }),
            ]),
            stored_output: Some(StoredOutput::Chat(vec![
                ContentBlockChatOutput::Text(Text {
                    text: "Test output 1".to_string(),
                }),
                ContentBlockChatOutput::Text(Text {
                    text: "Test output 2".to_string(),
                }),
            ])),
            dispreferred_outputs: vec![],
            episode_id: Some(episode_id),
            inference_id: Some(inference_id),
            tool_params: DynamicToolParams::default(),
            output_schema: None,
            tags: {
                let mut tags = HashMap::new();
                tags.insert("key1".to_string(), "value1".to_string());
                tags.insert("key2".to_string(), "value2".to_string());
                tags
            },
        }
    }

    /// Helper to create a test RenderedSample for JSON function
    fn create_test_json_rendered_sample() -> RenderedSample {
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();

        RenderedSample {
            function_name: "json_function".to_string(),
            function_type: FunctionType::Json,
            input: ModelInput {
                system: Some("JSON system prompt".to_string()),
                messages: vec![],
            },
            stored_input: StoredInput {
                system: Some(System::Text("JSON system prompt".to_string())),
                messages: vec![],
            },
            output: None, // JSON functions don't have chat output
            stored_output: Some(StoredOutput::Json(JsonInferenceOutput {
                raw: Some(r#"{"result": "test"}"#.to_string()),
                parsed: Some(serde_json::json!({"result": "test"})),
            })),
            dispreferred_outputs: vec![],
            episode_id: Some(episode_id),
            inference_id: Some(inference_id),
            tool_params: DynamicToolParams::default(),
            output_schema: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "result": {"type": "string"}
                }
            })),
            tags: {
                let mut tags = HashMap::new();
                tags.insert("json_key".to_string(), "json_value".to_string());
                tags
            },
        }
    }

    #[test]
    fn test_stored_inference_id() {
        let chat_inference = create_test_chat_inference();
        let json_inference = create_test_json_inference();

        let chat_id = StoredInference::Chat(chat_inference.clone()).id();
        let json_id = StoredInference::Json(json_inference.clone()).id();

        assert_eq!(chat_id, chat_inference.inference_id);
        assert_eq!(json_id, json_inference.inference_id);
    }

    // Tests for RenderedSample::into_create_datapoint_request()

    #[test]
    fn test_stored_sample_returns_none_for_missing_input() {
        let chat_inference = StoredChatInferenceDatabase {
            error: None,
            function_name: "test_function".to_string(),
            variant_name: "test_variant".to_string(),
            input: None,
            output: None,
            dispreferred_outputs: vec![],
            timestamp: Utc::now(),
            episode_id: Uuid::now_v7(),
            inference_id: Uuid::now_v7(),
            tool_params: None,
            tags: HashMap::new(),
            extra_body: None,
            inference_params: None,
            processing_time_ms: None,
            ttft_ms: None,
            snapshot_hash: None,
        };
        let chat_db = StoredInferenceDatabase::Chat(chat_inference);
        assert!(
            chat_db.input().is_none(),
            "input() should return None when the stored input is missing"
        );
        assert!(
            chat_db.into_input().is_none(),
            "into_input() should return None when the stored input is missing"
        );

        let json_inference = StoredJsonInference {
            error: None,
            function_name: "json_function".to_string(),
            variant_name: "json_variant".to_string(),
            input: None,
            output: None,
            dispreferred_outputs: vec![],
            timestamp: Utc::now(),
            episode_id: Uuid::now_v7(),
            inference_id: Uuid::now_v7(),
            output_schema: None,
            tags: HashMap::new(),
            extra_body: None,
            inference_params: None,
            processing_time_ms: None,
            ttft_ms: None,
            snapshot_hash: None,
        };
        let json_db = StoredInferenceDatabase::Json(json_inference);
        assert!(
            json_db.input().is_none(),
            "input() should return None when the stored input is missing"
        );
        assert!(
            json_db.into_input().is_none(),
            "into_input() should return None when the stored input is missing"
        );
    }

    // ── Serde roundtrip tests with None data fields ──────────────────────

    #[test]
    fn test_chat_inference_serde_roundtrip_with_none_data() {
        let mut inference = create_test_chat_inference();
        inference.input = None;
        inference.output = None;
        inference.extra_body = None;
        inference.inference_params = None;

        let json = serde_json::to_value(&inference).unwrap();

        assert!(
            !json.as_object().unwrap().contains_key("input"),
            "None input should be omitted from serialized JSON"
        );
        assert!(
            !json.as_object().unwrap().contains_key("output"),
            "None output should be omitted from serialized JSON"
        );

        let deserialized: StoredChatInference =
            serde_json::from_value(json).expect("should deserialize back");
        assert_eq!(
            deserialized.input, None,
            "Deserialized input should be None"
        );
        assert_eq!(
            deserialized.output, None,
            "Deserialized output should be None"
        );
        assert_eq!(
            deserialized.extra_body, None,
            "Deserialized extra_body should be None"
        );
        assert_eq!(
            deserialized.inference_params, None,
            "Deserialized inference_params should be None"
        );
    }

    #[test]
    fn test_json_inference_serde_roundtrip_with_none_data() {
        let mut inference = create_test_json_inference();
        inference.input = None;
        inference.output = None;
        inference.output_schema = None;
        inference.extra_body = None;
        inference.inference_params = None;

        let json = serde_json::to_value(&inference).unwrap();

        assert!(
            !json.as_object().unwrap().contains_key("input"),
            "None input should be omitted from serialized JSON"
        );
        assert!(
            !json.as_object().unwrap().contains_key("output"),
            "None output should be omitted from serialized JSON"
        );
        assert!(
            !json.as_object().unwrap().contains_key("output_schema"),
            "None output_schema should be omitted from serialized JSON"
        );

        let deserialized: StoredJsonInference =
            serde_json::from_value(json).expect("should deserialize back");
        assert_eq!(
            deserialized.input, None,
            "Deserialized input should be None"
        );
        assert_eq!(
            deserialized.output, None,
            "Deserialized output should be None"
        );
        assert_eq!(
            deserialized.output_schema, None,
            "Deserialized output_schema should be None"
        );
        assert_eq!(
            deserialized.extra_body, None,
            "Deserialized extra_body should be None"
        );
        assert_eq!(
            deserialized.inference_params, None,
            "Deserialized inference_params should be None"
        );
    }

    // ── into_datapoint_insert error tests for missing input ──────────────

    // ── into_datapoint_insert error test for missing output_schema (JSON only) ──

    // ── into_datapoint_insert success with output_source=None and missing data ──

    // ── owned_simple_info propagates None correctly ──────────────────────

    #[test]
    fn test_owned_simple_info_chat_with_none_data() {
        let chat_db = StoredInferenceDatabase::Chat(StoredChatInferenceDatabase {
            error: None,
            function_name: "test_function".to_string(),
            variant_name: "test_variant".to_string(),
            input: None,
            output: None,
            dispreferred_outputs: vec![],
            timestamp: Utc::now(),
            episode_id: Uuid::now_v7(),
            inference_id: Uuid::now_v7(),
            tool_params: None,
            tags: HashMap::new(),
            extra_body: None,
            inference_params: None,
            processing_time_ms: None,
            ttft_ms: None,
            snapshot_hash: None,
        });

        let info = chat_db.owned_simple_info();
        assert_eq!(info.input, None, "input should be None");
        assert_eq!(info.output, None, "output should be None");
        assert_eq!(info.stored_output, None, "stored_output should be None");
        assert_eq!(
            info.function_type,
            FunctionType::Chat,
            "function_type should be Chat"
        );
    }

    #[test]
    fn test_owned_simple_info_json_with_none_data() {
        let json_db = StoredInferenceDatabase::Json(StoredJsonInference {
            error: None,
            function_name: "json_function".to_string(),
            variant_name: "json_variant".to_string(),
            input: None,
            output: None,
            dispreferred_outputs: vec![],
            timestamp: Utc::now(),
            episode_id: Uuid::now_v7(),
            inference_id: Uuid::now_v7(),
            output_schema: None,
            tags: HashMap::new(),
            extra_body: None,
            inference_params: None,
            processing_time_ms: None,
            ttft_ms: None,
            snapshot_hash: None,
        });

        let info = json_db.owned_simple_info();
        assert_eq!(info.input, None, "input should be None");
        assert_eq!(info.output, None, "output should be None");
        assert_eq!(info.stored_output, None, "stored_output should be None");
        assert_eq!(info.output_schema, None, "output_schema should be None");
        assert_eq!(
            info.function_type,
            FunctionType::Json,
            "function_type should be Json"
        );
    }
}

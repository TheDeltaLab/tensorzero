use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tensorzero_core::config::UninitializedVariantInfo;
use tensorzero_derive::TensorZeroDeserialize;

/// Represents a targeted edit operation to apply to a TensorZero config.
#[derive(ts_rs::TS, Clone, Debug, Serialize, TensorZeroDeserialize, JsonSchema)]
#[ts(export)]
#[serde(tag = "operation")]
#[serde(rename_all = "snake_case")]
pub enum EditPayload {
    UpsertVariant(Box<UpsertVariantPayload>),
}

#[derive(ts_rs::TS, Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[ts(export)]
pub struct UpsertVariantPayload {
    pub function_name: String,
    pub variant_name: String,
    pub variant: UninitializedVariantInfo,
}


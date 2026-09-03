// Modified by Delta-AI under Apache 2.0
use serde::{Deserialize, Serialize};

use crate::config::gateway::{
    AsyncInferenceConfig, AuthConfig, DashboardUiConfig, MetricsConfig, UninitializedGatewayConfig,
};
use crate::config::{ExportConfig, TemplateFilesystemAccess, UninitializedRelayConfig};

use super::cache_config::StoredCacheConfig;
use super::observability_config::StoredObservabilityConfig;

/// Stored version of `UninitializedGatewayConfig`.
///
/// Omits `deny_unknown_fields` and uses `Stored*` sub-types for nested configs
/// that may gain new fields across versions.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct StoredGatewayConfig {
    pub bind_address: Option<std::net::SocketAddr>,
    #[serde(default)]
    pub observability: StoredObservabilityConfig,
    #[serde(default)]
    pub debug: bool,
    #[serde(default)]
    pub template_filesystem_access: Option<TemplateFilesystemAccess>,
    #[serde(default)]
    pub export: ExportConfig,
    pub base_path: Option<String>,
    #[serde(default)]
    pub unstable_disable_feedback_target_validation: bool,
    #[serde(default)]
    pub unstable_error_json: bool,
    #[serde(default)]
    pub disable_pseudonymous_usage_analytics: bool,
    pub fetch_and_encode_input_files_before_inference: Option<bool>,
    #[serde(default)]
    pub auth: AuthConfig,
    pub global_outbound_http_timeout_ms: Option<u64>,
    pub global_outbound_http_intra_stream_read_timeout_ms: Option<u64>,
    #[serde(default)]
    pub relay: Option<UninitializedRelayConfig>,
    #[serde(default)]
    pub metrics: MetricsConfig,
    #[serde(default)]
    pub cache: StoredCacheConfig,
    #[serde(default, skip_serializing_if = "DashboardUiConfig::is_empty")]
    pub ui: DashboardUiConfig,
    #[serde(default, skip_serializing_if = "async_inference_config_is_default")]
    pub async_inference: AsyncInferenceConfig,
}

fn async_inference_config_is_default(config: &AsyncInferenceConfig) -> bool {
    config == &AsyncInferenceConfig::default()
}

impl From<UninitializedGatewayConfig> for StoredGatewayConfig {
    fn from(config: UninitializedGatewayConfig) -> Self {
        let UninitializedGatewayConfig {
            bind_address,
            observability,
            debug,
            template_filesystem_access,
            export,
            base_path,
            unstable_disable_feedback_target_validation,
            unstable_error_json,
            disable_pseudonymous_usage_analytics,
            fetch_and_encode_input_files_before_inference,
            auth,
            global_outbound_http_timeout_ms,
            global_outbound_http_intra_stream_read_timeout_ms,
            relay,
            metrics,
            cache,
            ui,
            async_inference,
        } = config;
        Self {
            bind_address,
            observability: observability.unwrap_or_default().into(),
            debug: debug.unwrap_or_default(),
            template_filesystem_access,
            export: export.unwrap_or_default(),
            base_path,
            unstable_disable_feedback_target_validation:
                unstable_disable_feedback_target_validation.unwrap_or_default(),
            unstable_error_json: unstable_error_json.unwrap_or_default(),
            disable_pseudonymous_usage_analytics: disable_pseudonymous_usage_analytics
                .unwrap_or_default(),
            fetch_and_encode_input_files_before_inference,
            auth: auth.unwrap_or_default(),
            global_outbound_http_timeout_ms,
            global_outbound_http_intra_stream_read_timeout_ms,
            relay,
            metrics: metrics.unwrap_or_default(),
            cache: cache.unwrap_or_default().into(),
            ui: ui.unwrap_or_default(),
            async_inference: async_inference.unwrap_or_default(),
        }
    }
}

impl From<StoredGatewayConfig> for UninitializedGatewayConfig {
    fn from(stored: StoredGatewayConfig) -> Self {
        let StoredGatewayConfig {
            bind_address,
            observability,
            debug,
            template_filesystem_access,
            export,
            base_path,
            unstable_disable_feedback_target_validation,
            unstable_error_json,
            disable_pseudonymous_usage_analytics,
            fetch_and_encode_input_files_before_inference,
            auth,
            global_outbound_http_timeout_ms,
            global_outbound_http_intra_stream_read_timeout_ms,
            relay,
            metrics,
            cache,
            ui,
            async_inference,
        } = stored;
        Self {
            bind_address,
            observability: Some(observability.into()),
            debug: Some(debug),
            template_filesystem_access,
            export: Some(export),
            base_path,
            unstable_disable_feedback_target_validation: Some(
                unstable_disable_feedback_target_validation,
            ),
            unstable_error_json: Some(unstable_error_json),
            disable_pseudonymous_usage_analytics: Some(disable_pseudonymous_usage_analytics),
            fetch_and_encode_input_files_before_inference,
            auth: Some(auth),
            global_outbound_http_timeout_ms,
            global_outbound_http_intra_stream_read_timeout_ms,
            relay,
            metrics: Some(metrics),
            cache: Some(cache.into()),
            ui: if ui.is_empty() { None } else { Some(ui) },
            async_inference: if async_inference_config_is_default(&async_inference) {
                None
            } else {
                Some(async_inference)
            },
        }
    }
}

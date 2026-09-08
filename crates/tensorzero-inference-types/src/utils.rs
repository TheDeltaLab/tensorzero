// Modified by Delta-AI under Apache 2.0
//! Small utilities shared between providers and core.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use url::Url;

/// Emits a deprecation warning.
/// All deprecation warnings should be emitted using this function so that we can detect
/// unintentional use of deprecated behavior in our e2e tests.
pub fn deprecation_warning(message: &str) {
    tracing::warn!("Deprecation Warning: {message}");
}

/// Returns true if we're in mock mode (`TENSORZERO_INTERNAL_MOCK_PROVIDER_API` is set and non-empty).
pub fn is_mock_mode() -> bool {
    std::env::var("TENSORZERO_INTERNAL_MOCK_PROVIDER_API")
        .ok()
        .filter(|s| !s.is_empty())
        .is_some()
}

/// Returns the mock API base URL with the provider suffix appended.
/// Reads from `TENSORZERO_INTERNAL_MOCK_PROVIDER_API` env var.
/// Handles trailing slash normalization. Maps empty string to None.
pub fn get_mock_provider_api_base(provider_suffix: &str) -> Option<Url> {
    std::env::var("TENSORZERO_INTERNAL_MOCK_PROVIDER_API")
        .ok()
        .filter(|s| !s.is_empty())
        .and_then(|base| {
            let needs_slash = !base.ends_with('/')
                && !provider_suffix.starts_with('/')
                && !provider_suffix.is_empty();
            let base = if needs_slash {
                format!("{base}/")
            } else {
                base
            };
            Url::parse(&format!("{base}{provider_suffix}")).ok()
        })
}

/// Returns the current timestamp in seconds since the Unix epoch.
#[expect(clippy::missing_panics_doc)]
pub fn current_timestamp() -> u64 {
    #[expect(clippy::expect_used)]
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_secs()
}

/// Serializes a value that implements `Serialize` into a JSON string.
/// If serialization fails, it logs the error and returns an empty string.
///
/// # Arguments
///
/// * `value` - A reference to the value to be serialized.
///
/// # Returns
///
/// A `String` containing the serialized JSON, or an empty string if serialization fails.
pub fn serialize_or_log<T: Serialize>(value: &T) -> String {
    match serde_json::to_string(value) {
        Ok(serialized) => serialized,
        Err(e) => {
            tracing::error!("Failed to serialize value: {e}");
            String::new()
        }
    }
}

/// Warns that a provider does not support an inference parameter, so it will be ignored.
pub fn warn_inference_parameter_not_supported(
    model_provider_name: &str,
    parameter_name: &str,
    suffix: Option<&str>,
) {
    let mut message = format!(
        "{model_provider_name} does not support the inference parameter `{parameter_name}`, so it will be ignored."
    );
    if let Some(suffix) = suffix {
        message.push_str(&format!(" {suffix}"));
    }
    tracing::warn!("{}", message);
}

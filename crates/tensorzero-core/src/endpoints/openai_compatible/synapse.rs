// Modified by Delta-AI under Apache 2.0
//! Synapse compatibility helpers for OpenAI-compatible endpoints.
//!
//! Callers that previously hit Synapse send bare model names, `x-synapse-*`
//! request headers, and expect `x-synapse-served-by` / `x-synapse-request-id`
//! on the response. TensorZero also accepts the canonical `x-tensorzero-*`
//! names (preferred when both are set) and returns both prefixes so Trinity /
//! Lovelace / Cortex keep working while new callers can use TensorZero headers.

use std::collections::HashMap;
use std::time::Duration;

use axum::http::{HeaderMap, HeaderName, HeaderValue};
use axum::response::Response;
use uuid::Uuid;

use crate::cache::{CacheEnabledMode, CacheParamsOptions};
use crate::endpoints::inference::Params;
use crate::endpoints::openai_compatible::stream_aggregator::{
    StreamAggregateRule, parse_stream_aggregate_header,
};
use crate::error::{Error, ErrorDetails, TimeoutKind};
use crate::http::scope_request_timeout;
use crate::observability_tags::{
    HEADER_TAG_PREFIX, REQUEST_HEADERS_TAG, SYNAPSE_REQUEST_ID_TAG, inbound_request_headers,
    is_valid_request_id,
};

pub use crate::observability_tags::{
    TENSORZERO_EPISODE_ID_HEADER, TENSORZERO_EPISODES_ID_HEADER, TENSORZERO_TAGS_HEADER,
    overlay_compat_headers,
};

pub const SYNAPSE_PROVIDER_HEADER: &str = "x-synapse-provider";
pub const SYNAPSE_FALLBACK_HEADER: &str = "x-synapse-fallback";
pub const SYNAPSE_REQUEST_ID_HEADER: &str = "x-synapse-request-id";
pub const SYNAPSE_SERVED_BY_HEADER: &str = "x-synapse-served-by";
pub const SYNAPSE_FALLBACK_COUNT_HEADER: &str = "x-synapse-fallback-count";
pub const SYNAPSE_CACHE_HEADER: &str = "x-synapse-cache";
pub const SYNAPSE_STREAM_AGGREGATE_HEADER: &str = "x-synapse-stream-aggregate";
pub const SYNAPSE_REQUEST_PROFILE_HEADER: &str = "x-synapse-request-profile";
pub const SYNAPSE_RESPONSE_STYLE_HEADER: &str = "x-synapse-response-style";
pub const TENSORZERO_PROVIDER_HEADER: &str = "x-tensorzero-provider";
pub const TENSORZERO_FALLBACK_HEADER: &str = "x-tensorzero-fallback";
pub const TENSORZERO_REQUEST_ID_HEADER: &str = "x-tensorzero-request-id";
pub const TENSORZERO_SERVED_BY_HEADER: &str = "x-tensorzero-served-by";
pub const TENSORZERO_FALLBACK_COUNT_HEADER: &str = "x-tensorzero-fallback-count";
pub const TENSORZERO_CACHE_HEADER: &str = "x-tensorzero-cache";
pub const TENSORZERO_STREAM_AGGREGATE_HEADER: &str = "x-tensorzero-stream-aggregate";
pub const TENSORZERO_REQUEST_PROFILE_HEADER: &str = "x-tensorzero-request-profile";
pub const TENSORZERO_RESPONSE_STYLE_HEADER: &str = "x-tensorzero-response-style";
const X_REQUEST_ID_HEADER: &str = "x-request-id";
const LONG_AUDIO_EVAL_PROFILE: &str = "long-audio-eval";
pub const LONG_AUDIO_EVAL_TIMEOUT: Duration = Duration::from_secs(25 * 60);

static SYNAPSE_REQUEST_ID: HeaderName = HeaderName::from_static("x-synapse-request-id");
static SYNAPSE_SERVED_BY: HeaderName = HeaderName::from_static("x-synapse-served-by");
static SYNAPSE_FALLBACK_COUNT: HeaderName = HeaderName::from_static("x-synapse-fallback-count");
static TENSORZERO_REQUEST_ID: HeaderName = HeaderName::from_static("x-tensorzero-request-id");
static TENSORZERO_SERVED_BY: HeaderName = HeaderName::from_static("x-tensorzero-served-by");
static TENSORZERO_FALLBACK_COUNT: HeaderName =
    HeaderName::from_static("x-tensorzero-fallback-count");

/// Request-scoped Synapse compatibility state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SynapseRequestContext {
    pub request_id: String,
    pub provider: Option<String>,
    /// `x-tensorzero-fallback` / `x-synapse-fallback: false` disables alias /
    /// routing fallback for this request (head candidate only). Pinning via
    /// `x-tensorzero-provider` / `x-synapse-provider` already selects a single
    /// shorthand, which is then rotated to the front of any matching alias
    /// chain unless fallback is disabled.
    pub fallback_disabled: bool,
    pub served_by: Option<String>,
    pub fallback_count: u32,
    /// `x-tensorzero-cache` / `x-synapse-cache: false` skips cache read and write.
    pub cache_disabled: bool,
    /// Parsed `x-tensorzero-stream-aggregate` / `x-synapse-stream-aggregate`
    /// rules. `None` means pass through.
    pub stream_aggregate: Option<Vec<StreamAggregateRule>>,
    /// Per-request upstream timeout from `x-tensorzero-request-profile` /
    /// `x-synapse-request-profile`.
    pub request_timeout: Option<Duration>,
    /// `x-tensorzero-response-style` / `x-synapse-response-style: anthropic`
    /// on an OpenAI path.
    pub response_style_anthropic: bool,
}

impl SynapseRequestContext {
    pub fn from_headers(headers: &HeaderMap) -> Self {
        Self::try_from_headers(headers).unwrap_or_else(|_| Self {
            request_id: Uuid::now_v7().to_string(),
            provider: None,
            fallback_disabled: false,
            served_by: None,
            fallback_count: 0,
            cache_disabled: false,
            stream_aggregate: None,
            request_timeout: None,
            response_style_anthropic: false,
        })
    }

    pub fn try_from_headers(headers: &HeaderMap) -> Result<Self, Error> {
        let provider = compat_header(headers, TENSORZERO_PROVIDER_HEADER, SYNAPSE_PROVIDER_HEADER)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned);
        let fallback_disabled =
            header_is_false(headers, TENSORZERO_FALLBACK_HEADER, SYNAPSE_FALLBACK_HEADER);
        let cache_disabled =
            header_is_false(headers, TENSORZERO_CACHE_HEADER, SYNAPSE_CACHE_HEADER);
        let response_style_anthropic = compat_header(
            headers,
            TENSORZERO_RESPONSE_STYLE_HEADER,
            SYNAPSE_RESPONSE_STYLE_HEADER,
        )
        .is_some_and(|value| value.eq_ignore_ascii_case("anthropic"));
        let stream_aggregate = match compat_header(
            headers,
            TENSORZERO_STREAM_AGGREGATE_HEADER,
            SYNAPSE_STREAM_AGGREGATE_HEADER,
        ) {
            Some(raw) if !raw.trim().is_empty() => Some(parse_stream_aggregate_header(raw)?),
            _ => None,
        };
        let request_timeout = match compat_header(
            headers,
            TENSORZERO_REQUEST_PROFILE_HEADER,
            SYNAPSE_REQUEST_PROFILE_HEADER,
        ) {
            Some(raw) if !raw.trim().is_empty() => Some(parse_request_profile(raw)?),
            _ => None,
        };
        let request_id = inbound_request_id(headers).unwrap_or_else(|| Uuid::now_v7().to_string());
        Ok(Self {
            request_id,
            provider,
            fallback_disabled,
            served_by: None,
            fallback_count: 0,
            cache_disabled,
            stream_aggregate,
            request_timeout,
            response_style_anthropic,
        })
    }

    pub fn with_served_by_from_params(mut self, params: &Params) -> Self {
        if let Some(model_name) = &params.model_name {
            self.served_by = Some(served_by_from_model_name(model_name));
        } else if let Some(function_name) = &params.function_name {
            self.served_by = Some(function_name.clone());
        }
        self
    }

    pub fn apply_to_response(&self, response: &mut Response) {
        insert_header(response, &SYNAPSE_REQUEST_ID, &self.request_id);
        insert_header(response, &TENSORZERO_REQUEST_ID, &self.request_id);
        if let Some(served_by) = &self.served_by {
            insert_header(response, &SYNAPSE_SERVED_BY, served_by);
            insert_header(response, &TENSORZERO_SERVED_BY, served_by);
        }
        let fallback_count = self.fallback_count.to_string();
        insert_header(response, &SYNAPSE_FALLBACK_COUNT, &fallback_count);
        insert_header(response, &TENSORZERO_FALLBACK_COUNT, &fallback_count);
    }

    /// Tags written onto the inference row so logs can be queried by request
    /// id / provider / inbound Synapse headers (not Authorization).
    pub fn observability_tags(&self, headers: &HeaderMap) -> HashMap<String, String> {
        let mut tags = HashMap::new();
        tags.insert(SYNAPSE_REQUEST_ID_TAG.to_string(), self.request_id.clone());
        if let Some(provider) = &self.provider {
            tags.insert(
                crate::observability_tags::PROVIDER_TAG.to_string(),
                provider.clone(),
            );
        }
        if let Some(served_by) = &self.served_by {
            tags.insert(
                crate::observability_tags::SERVED_BY_TAG.to_string(),
                served_by.clone(),
            );
        }
        let request_headers = inbound_request_headers(headers);
        for (name, value) in &request_headers {
            tags.insert(format!("{HEADER_TAG_PREFIX}{name}"), value.clone());
        }
        if !request_headers.is_empty()
            && let Ok(json) = serde_json::to_string(&request_headers)
        {
            tags.insert(REQUEST_HEADERS_TAG.to_string(), json);
        }
        crate::observability_tags::insert_api_key_public_id_from_headers(&mut tags, headers);
        tags
    }
}

/// Overlay episode id and user tags from request headers. Body fields win.
pub fn apply_compat_to_params(headers: &HeaderMap, params: &mut Params) -> Result<(), Error> {
    overlay_compat_headers(headers, &mut params.episode_id, &mut params.tags)
}

/// Apply `x-tensorzero-provider` / `x-synapse-provider` to a (possibly bare) model name.
///
/// A provider header rewrites `gpt-4o` into `openai::gpt-4o`. If a matching
/// alias contains that `(provider, model)` pair, lookup rotates it to the
/// front of the failover chain (unless `x-synapse-fallback: false`). Names
/// that already contain `::` (shorthand or `tensorzero::…`) are left unchanged.
pub fn resolve_openai_compatible_model(
    model: &str,
    provider: Option<&str>,
) -> Result<String, Error> {
    let model = model.trim();
    if model.is_empty() {
        return Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
            message: "`model` field must not be empty".to_string(),
        }));
    }
    match provider.map(str::trim).filter(|value| !value.is_empty()) {
        Some(provider) if !model.contains("::") => Ok(format!("{provider}::{model}")),
        _ => Ok(model.to_string()),
    }
}

/// Default cache ON for Synapse-compatible endpoints unless the caller
/// set `tensorzero::cache_options` or `x-synapse-cache: false`.
pub fn resolve_cache_options(
    explicit: Option<CacheParamsOptions>,
    cache_disabled: bool,
) -> CacheParamsOptions {
    if cache_disabled {
        return CacheParamsOptions {
            max_age_s: None,
            enabled: CacheEnabledMode::Off,
        };
    }
    explicit.unwrap_or(CacheParamsOptions {
        max_age_s: None,
        enabled: CacheEnabledMode::On,
    })
}

pub fn parse_request_profile(raw: &str) -> Result<Duration, Error> {
    let profile = raw.trim();
    if profile.eq_ignore_ascii_case(LONG_AUDIO_EVAL_PROFILE) {
        return Ok(LONG_AUDIO_EVAL_TIMEOUT);
    }
    Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
        message: format!(
            "Unsupported request profile `{profile}` (`x-tensorzero-request-profile` / `x-synapse-request-profile`)"
        ),
    }))
}

pub async fn run_with_request_timeout<F, T>(timeout: Option<Duration>, fut: F) -> Result<T, Error>
where
    F: std::future::Future<Output = Result<T, Error>>,
{
    let Some(timeout) = timeout else {
        return fut.await;
    };
    scope_request_timeout(timeout, async {
        tokio::time::timeout(timeout, fut)
            .await
            .unwrap_or_else(|_| {
                Err(Error::new(ErrorDetails::ModelTimeout {
                    model_name: "synapse-request-profile".to_string(),
                    timeout,
                    kind: TimeoutKind::NonStreamingTotal,
                }))
            })
    })
    .await
}

/// Synapse `x-synapse-served-by` is `provider/model`. TensorZero shorthand
/// names use `provider::model`.
pub fn served_by_from_model_name(model_name: &str) -> String {
    match model_name.split_once("::") {
        Some((provider, rest)) => format!("{provider}/{rest}"),
        None => model_name.to_string(),
    }
}

fn inbound_request_id(headers: &HeaderMap) -> Option<String> {
    compat_header(
        headers,
        TENSORZERO_REQUEST_ID_HEADER,
        SYNAPSE_REQUEST_ID_HEADER,
    )
    .or_else(|| header_str(headers, X_REQUEST_ID_HEADER))
    .filter(|value| is_valid_request_id(value))
    .map(ToOwned::to_owned)
}

/// Prefer `x-tensorzero-*` when both prefixes are present.
fn compat_header<'a>(headers: &'a HeaderMap, tensorzero: &str, synapse: &str) -> Option<&'a str> {
    header_str(headers, tensorzero).or_else(|| header_str(headers, synapse))
}

fn header_is_false(headers: &HeaderMap, tensorzero: &str, synapse: &str) -> bool {
    compat_header(headers, tensorzero, synapse)
        .is_some_and(|value| value.eq_ignore_ascii_case("false"))
}

fn header_str<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name).and_then(|value| value.to_str().ok())
}

fn insert_header(response: &mut Response, name: &'static HeaderName, value: &str) {
    if let Ok(header_value) = HeaderValue::from_str(value) {
        response.headers_mut().insert(name.clone(), header_value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderMap;
    use googletest::prelude::*;

    #[gtest]
    fn test_resolve_bare_model_with_provider() {
        expect_that!(
            resolve_openai_compatible_model("deepseek-v4-flash", Some("deepseek")).unwrap(),
            eq("deepseek::deepseek-v4-flash")
        );
    }

    #[test]
    fn test_resolve_already_qualified_model_ignores_provider() {
        assert_eq!(
            resolve_openai_compatible_model("deepseek::deepseek-v4-flash", Some("openai")).unwrap(),
            "deepseek::deepseek-v4-flash"
        );
        assert_eq!(
            resolve_openai_compatible_model("tensorzero::model_name::my_model", Some("openai"))
                .unwrap(),
            "tensorzero::model_name::my_model"
        );
    }

    #[test]
    fn test_resolve_empty_model_errors() {
        let err = resolve_openai_compatible_model("  ", None).unwrap_err();
        assert!(err.to_string().contains("`model` field must not be empty"));
    }

    #[test]
    fn test_served_by_from_shorthand() {
        assert_eq!(
            served_by_from_model_name("deepseek::deepseek-v4-pro"),
            "deepseek/deepseek-v4-pro"
        );
        assert_eq!(served_by_from_model_name("my-alias"), "my-alias");
    }

    #[test]
    fn test_request_context_honors_inbound_id_and_fallback() {
        let mut headers = HeaderMap::new();
        headers.insert(SYNAPSE_REQUEST_ID_HEADER, "caller-trace-1".parse().unwrap());
        headers.insert(SYNAPSE_PROVIDER_HEADER, "volcengine".parse().unwrap());
        headers.insert(SYNAPSE_FALLBACK_HEADER, "false".parse().unwrap());
        let ctx = SynapseRequestContext::from_headers(&headers);
        assert_eq!(ctx.request_id, "caller-trace-1");
        assert_eq!(ctx.provider.as_deref(), Some("volcengine"));
        assert!(ctx.fallback_disabled);
    }

    #[test]
    fn test_request_context_falls_back_to_x_request_id() {
        let mut headers = HeaderMap::new();
        headers.insert(X_REQUEST_ID_HEADER, "std-trace".parse().unwrap());
        let ctx = SynapseRequestContext::from_headers(&headers);
        assert_eq!(ctx.request_id, "std-trace");
    }

    #[test]
    fn test_cache_disabled_and_request_profile() {
        let mut headers = HeaderMap::new();
        headers.insert(SYNAPSE_CACHE_HEADER, "false".parse().unwrap());
        headers.insert(
            SYNAPSE_REQUEST_PROFILE_HEADER,
            "long-audio-eval".parse().unwrap(),
        );
        let ctx = SynapseRequestContext::try_from_headers(&headers).unwrap();
        assert!(ctx.cache_disabled);
        assert_eq!(ctx.request_timeout, Some(LONG_AUDIO_EVAL_TIMEOUT));
        let opts = resolve_cache_options(None, true);
        assert_eq!(opts.enabled, CacheEnabledMode::Off);
        let on = resolve_cache_options(None, false);
        assert_eq!(on.enabled, CacheEnabledMode::On);
    }

    #[test]
    fn test_unknown_request_profile_errors() {
        let mut headers = HeaderMap::new();
        headers.insert(SYNAPSE_REQUEST_PROFILE_HEADER, "nope".parse().unwrap());
        assert!(SynapseRequestContext::try_from_headers(&headers).is_err());
    }

    #[test]
    fn test_invalid_stream_aggregate_errors() {
        let mut headers = HeaderMap::new();
        headers.insert(SYNAPSE_STREAM_AGGREGATE_HEADER, "[]".parse().unwrap());
        assert!(SynapseRequestContext::try_from_headers(&headers).is_err());
    }

    #[gtest]
    fn observability_tags_capture_request_id_and_headers() {
        let mut headers = HeaderMap::new();
        headers.insert(SYNAPSE_REQUEST_ID_HEADER, "caller-trace-1".parse().unwrap());
        headers.insert(SYNAPSE_PROVIDER_HEADER, "volcengine".parse().unwrap());
        headers.insert("authorization", "Bearer secret".parse().unwrap());
        let ctx = SynapseRequestContext::from_headers(&headers);
        let tags = ctx.observability_tags(&headers);
        expect_that!(
            tags.get(SYNAPSE_REQUEST_ID_TAG).map(String::as_str),
            eq(Some("caller-trace-1"))
        );
        expect_that!(
            tags.get(crate::observability_tags::PROVIDER_TAG)
                .map(String::as_str),
            eq(Some("volcengine"))
        );
        expect_that!(
            tags.get("tensorzero::header::x-tensorzero-request-id")
                .map(String::as_str),
            eq(Some("caller-trace-1"))
        );
        expect_that!(
            tags.values().any(|value| value.contains("secret")),
            eq(false)
        );
    }

    #[test]
    fn tensorzero_headers_win_over_synapse() {
        let mut headers = HeaderMap::new();
        headers.insert(TENSORZERO_PROVIDER_HEADER, "openai".parse().unwrap());
        headers.insert(SYNAPSE_PROVIDER_HEADER, "deepseek".parse().unwrap());
        headers.insert(TENSORZERO_CACHE_HEADER, "true".parse().unwrap());
        headers.insert(SYNAPSE_CACHE_HEADER, "false".parse().unwrap());
        headers.insert(TENSORZERO_REQUEST_ID_HEADER, "tz-id".parse().unwrap());
        headers.insert(SYNAPSE_REQUEST_ID_HEADER, "syn-id".parse().unwrap());
        let ctx = SynapseRequestContext::from_headers(&headers);
        assert_eq!(ctx.provider.as_deref(), Some("openai"));
        assert!(!ctx.cache_disabled);
        assert_eq!(ctx.request_id, "tz-id");
    }

    #[test]
    fn overlay_episode_and_tags_from_headers() {
        let episode = Uuid::now_v7();
        let mut headers = HeaderMap::new();
        headers.insert(
            TENSORZERO_EPISODES_ID_HEADER,
            episode.to_string().parse().unwrap(),
        );
        headers.insert(
            TENSORZERO_TAGS_HEADER,
            "env=prod,team=ml,canary".parse().unwrap(),
        );
        let mut episode_id = None;
        let mut tags = HashMap::from([("env".to_string(), "staging".to_string())]);
        overlay_compat_headers(&headers, &mut episode_id, &mut tags).unwrap();
        assert_eq!(episode_id, Some(episode));
        assert_eq!(tags.get("env").map(String::as_str), Some("staging"));
        assert_eq!(tags.get("team").map(String::as_str), Some("ml"));
        assert_eq!(tags.get("canary").map(String::as_str), Some("true"));
    }
}

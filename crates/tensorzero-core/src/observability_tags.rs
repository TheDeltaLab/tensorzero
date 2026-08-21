// Modified by Delta-AI under Apache 2.0
//! Observability tags stored on inference rows for Synapse-style log search.
//!
//! Inbound `x-tensorzero-*` / `x-synapse-*` / `x-request-id` headers and upstream
//! vendor request ids are written as `tensorzero::` tags so the Inferences UI
//! can filter by request id, provider, cache, and user tags.

use std::collections::HashMap;

use http::HeaderMap;
use rust_decimal::Decimal;
use tensorzero_auth::key::TensorZeroApiKey;
use uuid::Uuid;

use crate::error::{Error, ErrorDetails};

pub const SYNAPSE_REQUEST_ID_TAG: &str = "tensorzero::synapse_request_id";
pub const PROVIDER_REQUEST_ID_TAG: &str = "tensorzero::provider_request_id";
pub const PROVIDER_TAG: &str = "tensorzero::provider";
pub const SERVED_BY_TAG: &str = "tensorzero::served_by";
pub const CACHED_TAG: &str = "tensorzero::cached";
pub const FALLBACK_COUNT_TAG: &str = "tensorzero::fallback_count";
pub const PROVIDER_DEBUG_TAG: &str = "tensorzero::provider_debug";
pub const REQUEST_HEADERS_TAG: &str = "tensorzero::request_headers";
pub const HEADER_TAG_PREFIX: &str = "tensorzero::header::";
pub const INPUT_TOKENS_TAG: &str = "tensorzero::input_tokens";
pub const OUTPUT_TOKENS_TAG: &str = "tensorzero::output_tokens";
pub const COST_TAG: &str = "tensorzero::cost";
pub const CURRENCY_TAG: &str = "tensorzero::currency";
pub const STATUS_CODE_TAG: &str = "tensorzero::status_code";
pub const API_KEY_PUBLIC_ID_TAG: &str = "tensorzero::api_key_public_id";
pub const TENSORZERO_EPISODE_ID_HEADER: &str = "x-tensorzero-episode-id";
pub const TENSORZERO_EPISODES_ID_HEADER: &str = "x-tensorzero-episodes-id";
pub const TENSORZERO_TAGS_HEADER: &str = "x-tensorzero-tags";

const MAX_HEADER_VALUE_LEN: usize = 500;
const MAX_REQUEST_ID_LEN: usize = 200;
const MAX_USER_TAGS: usize = 50;
const MAX_TAG_KEY_LEN: usize = 128;

/// Inbound headers that have dedicated storage (episode id / user tags).
const SKIP_CAPTURE_HEADERS: &[&str] = &[
    "x-tensorzero-tags",
    "x-tensorzero-episode-id",
    "x-tensorzero-episodes-id",
];

const SENSITIVE_HEADER_NAMES: &[&str] = &[
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
    "api-key",
];

const PROVIDER_REQUEST_ID_HEADERS: &[&str] = &[
    "x-request-id",
    "request-id",
    "x-ds-request-id",
    "x-goog-request-id",
    "x-dashscope-request-id",
];

const PROVIDER_DEBUG_PREFIXES: &[&str] = &[
    "openai-",
    "x-ratelimit-",
    "anthropic-",
    "x-goog-",
    "x-or-",
    "x-ds-",
    "x-dashscope-",
];

const PROVIDER_DEBUG_EXACT: &[&str] = &["server-timing", "x-request-id", "request-id"];

/// Upstream vendor request id plus a small debug-header map (rate limits, etc.).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct UpstreamMetadata {
    pub provider_request_id: Option<String>,
    pub provider_debug: HashMap<String, String>,
}

pub fn extract_upstream_metadata(headers: &HeaderMap) -> UpstreamMetadata {
    let mut provider_request_id = None;
    for name in PROVIDER_REQUEST_ID_HEADERS {
        if let Some(value) = header_value(headers, name) {
            provider_request_id = Some(value);
            break;
        }
    }

    let mut provider_debug = HashMap::new();
    for (name, value) in headers {
        let lower = name.as_str().to_ascii_lowercase();
        if is_sensitive_header(&lower) {
            continue;
        }
        let keep = PROVIDER_DEBUG_EXACT.contains(&lower.as_str())
            || PROVIDER_DEBUG_PREFIXES
                .iter()
                .any(|prefix| lower.starts_with(prefix));
        if !keep {
            continue;
        }
        let Some(text) = value.to_str().ok().map(str::trim).filter(|s| !s.is_empty()) else {
            continue;
        };
        provider_debug.insert(lower, truncate(text).to_string());
    }

    UpstreamMetadata {
        provider_request_id,
        provider_debug,
    }
}

/// Safe inbound headers for a query: TensorZero / Synapse tracing headers and
/// `x-request-id`. Synapse names are stored under the `x-tensorzero-*` form so
/// the UI Headers section shows the canonical prefix. `x-tensorzero-*` wins
/// when both prefixes are present.
pub fn inbound_request_headers(headers: &HeaderMap) -> HashMap<String, String> {
    let mut captured = HashMap::new();
    for (name, value) in headers {
        let lower = name.as_str().to_ascii_lowercase();
        if !should_capture_inbound_header(&lower) {
            continue;
        }
        let Some(text) = value.to_str().ok().map(str::trim).filter(|s| !s.is_empty()) else {
            continue;
        };
        captured.insert(lower, truncate(text).to_string());
    }
    for (name, value) in headers {
        let lower = name.as_str().to_ascii_lowercase();
        let Some(rest) = lower.strip_prefix("x-synapse-") else {
            continue;
        };
        let canonical = format!("x-tensorzero-{rest}");
        if captured.contains_key(&canonical) || SKIP_CAPTURE_HEADERS.contains(&canonical.as_str()) {
            continue;
        }
        if is_sensitive_header(&lower) {
            continue;
        }
        let Some(text) = value.to_str().ok().map(str::trim).filter(|s| !s.is_empty()) else {
            continue;
        };
        captured.insert(canonical, truncate(text).to_string());
    }
    captured
}

/// Public id from `Authorization: Bearer` or `x-api-key` when the value is a
/// TensorZero (`sk-t0-…`) or Synapse (`sk-syn-v1-…`) key. Secrets are not stored.
pub fn api_key_public_id_from_headers(headers: &HeaderMap) -> Option<String> {
    let raw = raw_api_key_from_headers(headers)?;
    if TensorZeroApiKey::is_synapse_key(raw) {
        return Some(
            TensorZeroApiKey::from_synapse_plaintext(raw)
                .get_public_id()
                .to_string(),
        );
    }
    TensorZeroApiKey::parse(raw)
        .ok()
        .map(|key| key.get_public_id().to_string())
}

pub fn insert_api_key_public_id_from_headers(
    tags: &mut HashMap<String, String>,
    headers: &HeaderMap,
) {
    if tags.contains_key(API_KEY_PUBLIC_ID_TAG) {
        return;
    }
    if let Some(public_id) = api_key_public_id_from_headers(headers) {
        tags.insert(API_KEY_PUBLIC_ID_TAG.to_string(), public_id);
    }
}

fn raw_api_key_from_headers(headers: &HeaderMap) -> Option<&str> {
    if let Some(value) = headers.get(http::header::AUTHORIZATION) {
        let text = value.to_str().ok()?.trim();
        return text
            .strip_prefix("Bearer ")
            .map(str::trim)
            .filter(|value| !value.is_empty());
    }
    headers
        .get("x-api-key")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

fn should_capture_inbound_header(lower: &str) -> bool {
    if is_sensitive_header(lower) || SKIP_CAPTURE_HEADERS.contains(&lower) {
        return false;
    }
    lower == "x-request-id" || lower.starts_with("x-tensorzero-")
}

/// Parse `x-tensorzero-tags`: comma-separated `key=value` pairs (bare keys
/// become `true`). Body `tensorzero::tags` still win on key conflicts.
pub fn parse_csv_tags(raw: &str) -> Result<HashMap<String, String>, String> {
    let mut tags = HashMap::new();
    for piece in raw.split(',') {
        let piece = piece.trim();
        if piece.is_empty() {
            continue;
        }
        let (key, value) = match piece.split_once('=') {
            Some((key, value)) => (key.trim(), value.trim()),
            None => (piece, "true"),
        };
        if key.is_empty() {
            return Err("tag key must not be empty".to_string());
        }
        if key.len() > MAX_TAG_KEY_LEN {
            return Err(format!(
                "tag key `{key}` exceeds {MAX_TAG_KEY_LEN} characters"
            ));
        }
        if key.starts_with("tensorzero::") {
            return Err(format!("tag name cannot start with 'tensorzero::': {key}"));
        }
        if tags.len() >= MAX_USER_TAGS && !tags.contains_key(key) {
            return Err(format!("at most {MAX_USER_TAGS} tags are allowed"));
        }
        tags.insert(key.to_string(), truncate(value).to_string());
    }
    Ok(tags)
}

/// Overlay episode id and user tags from request headers. Body fields win.
pub fn overlay_compat_headers(
    headers: &HeaderMap,
    episode_id: &mut Option<Uuid>,
    tags: &mut HashMap<String, String>,
) -> Result<(), Error> {
    if episode_id.is_none() {
        *episode_id = parse_episode_id_header(headers)?;
    }
    for (key, value) in parse_tags_header(headers)? {
        tags.entry(key).or_insert(value);
    }
    Ok(())
}

pub fn parse_episode_id_header(headers: &HeaderMap) -> Result<Option<Uuid>, Error> {
    let raw = header_str(headers, TENSORZERO_EPISODE_ID_HEADER)
        .or_else(|| header_str(headers, TENSORZERO_EPISODES_ID_HEADER))
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let Some(raw) = raw else {
        return Ok(None);
    };
    Uuid::parse_str(raw).map(Some).map_err(|_| {
        Error::new(ErrorDetails::InvalidRequest {
            message: format!("`x-tensorzero-episode-id` must be a UUID, got `{raw}`"),
        })
    })
}

pub fn parse_tags_header(headers: &HeaderMap) -> Result<HashMap<String, String>, Error> {
    let Some(raw) = header_str(headers, TENSORZERO_TAGS_HEADER)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(HashMap::new());
    };
    parse_csv_tags(raw).map_err(|message| {
        Error::new(ErrorDetails::InvalidRequest {
            message: format!("Invalid `x-tensorzero-tags`: {message}"),
        })
    })
}

fn header_str<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name).and_then(|value| value.to_str().ok())
}

pub fn is_valid_request_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_REQUEST_ID_LEN
        && value.is_ascii()
        && !value.bytes().any(|byte| byte < 0x20 || byte == 0x7f)
}

fn is_sensitive_header(name: &str) -> bool {
    SENSITIVE_HEADER_NAMES.contains(&name)
}

fn header_value(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| is_valid_request_id(value))
        .map(ToOwned::to_owned)
}

fn truncate(value: &str) -> &str {
    if value.len() <= MAX_HEADER_VALUE_LEN {
        value
    } else {
        &value[..MAX_HEADER_VALUE_LEN]
    }
}

/// Persist usage so the Inferences list can show tokens/cost/status without
/// joining `model_inferences` for every row. Successful writes are `200`.
///
/// Cost and currency tags are omitted when any row is missing cost, or when
/// currencies disagree. Missing currency is treated as `USD`.
pub fn apply_usage_observability_tags<'a>(
    tags: &mut HashMap<String, String>,
    rows: impl IntoIterator<Item = (Option<u32>, Option<u32>, Option<Decimal>, Option<&'a str>)>,
) {
    tags.entry(STATUS_CODE_TAG.to_string())
        .or_insert_with(|| "200".to_string());

    let mut input_tokens = 0u64;
    let mut output_tokens = 0u64;
    let mut has_input = false;
    let mut has_output = false;
    let mut cost = Decimal::ZERO;
    let mut saw_row = false;
    let mut all_costs_present = true;
    let mut currency: Option<String> = None;
    let mut currencies_agree = true;

    for (input, output, row_cost, row_currency) in rows {
        saw_row = true;
        if let Some(tokens) = input {
            input_tokens += u64::from(tokens);
            has_input = true;
        }
        if let Some(tokens) = output {
            output_tokens += u64::from(tokens);
            has_output = true;
        }
        match row_cost {
            Some(value) => cost += value,
            None => all_costs_present = false,
        }
        let normalized = normalize_tag_currency(row_currency);
        match &currency {
            None => currency = Some(normalized),
            Some(existing) if existing == &normalized => {}
            Some(_) => currencies_agree = false,
        }
    }

    if has_input {
        tags.insert(INPUT_TOKENS_TAG.to_string(), input_tokens.to_string());
    }
    if has_output {
        tags.insert(OUTPUT_TOKENS_TAG.to_string(), output_tokens.to_string());
    }
    if saw_row && all_costs_present && currencies_agree {
        tags.insert(COST_TAG.to_string(), cost.normalize().to_string());
        if let Some(code) = currency {
            tags.insert(CURRENCY_TAG.to_string(), code);
        }
    }
}

fn normalize_tag_currency(value: Option<&str>) -> String {
    let code = value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("USD")
        .to_ascii_uppercase();
    if code == "RMB" {
        "CNY".to_string()
    } else {
        code
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use googletest::prelude::*;
    use http::HeaderValue;

    #[gtest]
    fn extracts_provider_request_id_and_debug_headers() {
        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", HeaderValue::from_static("req_openai_123"));
        headers.insert(
            "x-ratelimit-remaining-requests",
            HeaderValue::from_static("50"),
        );
        headers.insert("authorization", HeaderValue::from_static("Bearer secret"));
        headers.insert("unrelated", HeaderValue::from_static("nope"));
        let meta = extract_upstream_metadata(&headers);
        expect_that!(
            meta.provider_request_id.as_deref(),
            eq(Some("req_openai_123"))
        );
        expect_that!(
            meta.provider_debug
                .get("x-ratelimit-remaining-requests")
                .map(String::as_str),
            eq(Some("50"))
        );
        expect_that!(meta.provider_debug.contains_key("authorization"), eq(false));
        expect_that!(meta.provider_debug.contains_key("unrelated"), eq(false));
    }

    #[gtest]
    fn inbound_headers_keep_synapse_and_skip_secrets() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-synapse-request-id",
            HeaderValue::from_static("caller-trace-1"),
        );
        headers.insert("x-synapse-provider", HeaderValue::from_static("deepseek"));
        headers.insert("authorization", HeaderValue::from_static("Bearer secret"));
        headers.insert("x-request-id", HeaderValue::from_static("std-trace"));
        let captured = inbound_request_headers(&headers);
        expect_that!(
            captured.get("x-tensorzero-request-id").map(String::as_str),
            eq(Some("caller-trace-1"))
        );
        expect_that!(
            captured.get("x-tensorzero-provider").map(String::as_str),
            eq(Some("deepseek"))
        );
        expect_that!(
            captured.get("x-request-id").map(String::as_str),
            eq(Some("std-trace"))
        );
        expect_that!(captured.contains_key("authorization"), eq(false));
        expect_that!(captured.contains_key("x-synapse-request-id"), eq(false));
    }

    #[gtest]
    fn inbound_headers_prefer_tensorzero_prefix() {
        let mut headers = HeaderMap::new();
        headers.insert("x-tensorzero-provider", HeaderValue::from_static("openai"));
        headers.insert("x-synapse-provider", HeaderValue::from_static("deepseek"));
        headers.insert(
            "x-tensorzero-tags",
            HeaderValue::from_static("env=prod,team=ml"),
        );
        let captured = inbound_request_headers(&headers);
        expect_that!(
            captured.get("x-tensorzero-provider").map(String::as_str),
            eq(Some("openai"))
        );
        expect_that!(captured.contains_key("x-tensorzero-tags"), eq(false));
    }

    #[gtest]
    fn parse_csv_tags_supports_pairs_and_bare_keys() {
        let tags = parse_csv_tags("env=prod, team=ml, canary").unwrap();
        expect_that!(tags.get("env").map(String::as_str), eq(Some("prod")));
        expect_that!(tags.get("team").map(String::as_str), eq(Some("ml")));
        expect_that!(tags.get("canary").map(String::as_str), eq(Some("true")));
        expect_that!(
            parse_csv_tags("tensorzero::internal=true").is_err(),
            eq(true)
        );
    }

    #[gtest]
    fn usage_tags_sum_tokens_and_cost() {
        let mut tags = HashMap::new();
        apply_usage_observability_tags(
            &mut tags,
            [
                (Some(10), Some(4), Some(Decimal::new(12, 4)), Some("USD")),
                (Some(2), None, Some(Decimal::ZERO), None),
            ],
        );
        expect_that!(
            tags.get(STATUS_CODE_TAG).map(String::as_str),
            eq(Some("200"))
        );
        expect_that!(
            tags.get(INPUT_TOKENS_TAG).map(String::as_str),
            eq(Some("12"))
        );
        expect_that!(
            tags.get(OUTPUT_TOKENS_TAG).map(String::as_str),
            eq(Some("4"))
        );
        expect_that!(tags.get(COST_TAG).map(String::as_str), eq(Some("0.0012")));
        expect_that!(tags.get(CURRENCY_TAG).map(String::as_str), eq(Some("USD")));
    }

    #[gtest]
    fn usage_tags_omit_cost_when_any_row_missing() {
        let mut tags = HashMap::new();
        apply_usage_observability_tags(&mut tags, [(Some(1), Some(1), None, Some("USD"))]);
        expect_that!(tags.contains_key(COST_TAG), eq(false));
        expect_that!(tags.contains_key(CURRENCY_TAG), eq(false));
        expect_that!(
            tags.get(STATUS_CODE_TAG).map(String::as_str),
            eq(Some("200"))
        );
    }

    #[gtest]
    fn usage_tags_omit_cost_when_currencies_differ() {
        let mut tags = HashMap::new();
        apply_usage_observability_tags(
            &mut tags,
            [
                (Some(1), Some(1), Some(Decimal::ONE), Some("USD")),
                (Some(1), Some(1), Some(Decimal::ONE), Some("CNY")),
            ],
        );
        expect_that!(tags.contains_key(COST_TAG), eq(false));
        expect_that!(tags.contains_key(CURRENCY_TAG), eq(false));
    }

    #[gtest]
    fn usage_tags_treat_rmb_as_cny() {
        let mut tags = HashMap::new();
        apply_usage_observability_tags(
            &mut tags,
            [(Some(1), Some(1), Some(Decimal::new(6, 1)), Some("RMB"))],
        );
        expect_that!(tags.get(COST_TAG).map(String::as_str), eq(Some("0.6")));
        expect_that!(tags.get(CURRENCY_TAG).map(String::as_str), eq(Some("CNY")));
    }

    #[gtest]
    fn api_key_public_id_from_tensorzero_bearer() {
        let mut headers = HeaderMap::new();
        headers.insert(
            http::header::AUTHORIZATION,
            HeaderValue::from_static(
                "Bearer sk-t0-abcdefghijkl-123456789012345678901234567890123456789012345678",
            ),
        );
        expect_that!(
            api_key_public_id_from_headers(&headers).as_deref(),
            eq(Some("abcdefghijkl"))
        );
    }

    #[gtest]
    fn api_key_public_id_ignores_malformed_bearer() {
        let mut headers = HeaderMap::new();
        headers.insert(
            http::header::AUTHORIZATION,
            HeaderValue::from_static("Bearer secret"),
        );
        expect_that!(api_key_public_id_from_headers(&headers), none());
    }
}

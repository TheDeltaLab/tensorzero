// Modified by Delta-AI under Apache 2.0
//! OpenAI-compatible `POST /v1/rerank` for Synapse-compatible clients.
//!
//! Callers send `{ model, query, documents }` with `x-synapse-provider: alibaba`.
//! DashScope's compatible-api path is `/v1/reranks` (note the trailing `s`).
//!
//! A rerank `[model_aliases]` entry's `targets` are an ordered fallback chain:
//! candidates are tried head-first and the request fails over to the next
//! target (which may be a different provider AND a different model) on
//! network errors, timeouts, 401/402/403/408/429 and 5xx — the Synapse
//! `isFailoverableStatus` set. `x-synapse-fallback: false` keeps the head
//! candidate only. A provider-header pin (or `provider::model` shorthand)
//! rotates the matching alias target to the head of the chain, same as the
//! chat/embedding paths.
//!
//! `instruct` (task instruction) and `return_documents` are forwarded to the
//! upstream when present — DashScope rerank honors them, and providers without
//! support ignore unknown keys. Inbound `instruction` (flat or `parameters.*`)
//! is normalized to `instruct` because DashScope silently ignores `instruction`.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use axum::Json;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use url::Url;

use crate::cost::{ResponseMode, apply_computed_cost};
use crate::endpoints::standalone_inference::{
    RERANK_ENDPOINT, StandaloneInferenceRecord, StandaloneInput, maybe_write_standalone_inference,
    rerank_output_payload, usage_from_json,
};
use crate::error::{Error, ErrorDetails};
use crate::http::TensorzeroHttpClient;
use crate::inference::types::{Latency, Usage};
use crate::model::{SILICONFLOW_DEFAULT_API_ROOT, openai_compatible_shorthand_api_base};
use crate::model_alias::{ModelAlias, ModelAliasTable};
use crate::utils::gateway::{AppState, AppStateData};

use super::OpenAIStructuredJson;
use super::infer::error_response;
use super::synapse::{
    SynapseRequestContext, overlay_compat_headers, resolve_openai_compatible_model,
    run_with_request_timeout, served_by_from_model_name,
};

/// DashScope rerank is on `compatible-api`, not the chat `compatible-mode` host.
const ALIBABA_RERANK_DEFAULT_API_ROOT: &str = "https://dashscope.aliyuncs.com/compatible-api";

#[derive(Debug, Deserialize)]
pub struct OpenAICompatibleRerankParams {
    pub model: String,
    #[serde(default)]
    pub query: Option<String>,
    #[serde(default)]
    pub documents: Option<Vec<String>>,
    #[serde(default)]
    pub top_n: Option<u32>,
    /// DashScope-native: `{ input: { query, documents }, parameters: { top_n } }`
    #[serde(default)]
    pub input: Option<DashScopeRerankInput>,
    #[serde(default)]
    pub parameters: Option<DashScopeRerankParameters>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Debug, Deserialize)]
pub struct DashScopeRerankInput {
    pub query: Option<String>,
    pub documents: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct DashScopeRerankParameters {
    pub top_n: Option<u32>,
    pub return_documents: Option<bool>,
    pub instruct: Option<String>,
    /// Lenient alias for `instruct`; the real upstream param is `instruct` and
    /// DashScope silently ignores `instruction`, so we normalize to `instruct`.
    pub instruction: Option<String>,
}

/// Normalized rerank arguments extracted from either the Cohere-style flat
/// shape or the DashScope-style `input` / `parameters` shape.
#[derive(Debug)]
struct RerankArgs {
    query: String,
    documents: Vec<String>,
    top_n: Option<u32>,
    return_documents: Option<bool>,
    instruct: Option<String>,
}

#[derive(Debug, Serialize)]
struct CohereRerankResult {
    index: usize,
    relevance_score: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    document: Option<CohereRerankDocument>,
}

#[derive(Debug, Serialize)]
struct CohereRerankDocument {
    text: String,
}

pub async fn rerank_handler(
    State(AppStateData {
        http_client,
        config,
        clickhouse_connection_info,
        postgres_connection_info,
        deferred_tasks,
        ..
    }): AppState,
    headers: HeaderMap,
    OpenAIStructuredJson(params): OpenAIStructuredJson<OpenAICompatibleRerankParams>,
) -> Result<Response, crate::endpoints::openai_compatible::OpenAICompatibleError> {
    let mut synapse = match SynapseRequestContext::try_from_headers(&headers) {
        Ok(ctx) => ctx,
        Err(error) => {
            return Ok(error_response(
                error,
                false,
                &SynapseRequestContext::from_headers(&headers),
            ));
        }
    };
    let candidates = match resolve_rerank_candidates(
        &params.model,
        synapse.provider.as_deref(),
        &config.models.model_aliases,
        synapse.fallback_disabled,
    ) {
        Ok(candidates) => candidates,
        Err(error) => return Ok(error_response(error, false, &synapse)),
    };

    let args = match extract_rerank_args(&params) {
        Ok(args) => args,
        Err(error) => return Ok(error_response(error, false, &synapse)),
    };

    let start = Instant::now();
    let dispatch = dispatch_with_fallback(
        &http_client,
        &candidates,
        &args,
        &params.extra,
        synapse.request_timeout,
    )
    .await;
    let latency = Latency::NonStreaming {
        response_time: start.elapsed(),
    };

    let (status, mut body, provider_name, upstream_name, raw_request) = match dispatch {
        RerankDispatch::Served {
            index,
            provider_name,
            upstream_name,
            raw_request,
            status,
            body,
        } => {
            synapse.served_by = Some(served_by_from_model_name(&format!(
                "{provider_name}::{upstream_name}"
            )));
            synapse.fallback_count = u32::try_from(index).unwrap_or(u32::MAX);
            (status, body, provider_name, upstream_name, raw_request)
        }
        RerankDispatch::Error(error) => return Ok(error_response(error, false, &synapse)),
        RerankDispatch::Exhausted { failure } => return Ok(failure.into_response(&synapse)),
    };

    if status.is_success() {
        let RerankArgs {
            query, documents, ..
        } = args;
        let usage = apply_rerank_cost(&config, &provider_name, &upstream_name, &body);
        overlay_rerank_usage(&mut body, &usage);
        let mut episode_id = None;
        let mut tags = HashMap::new();
        if let Err(error) = overlay_compat_headers(&headers, &mut episode_id, &mut tags) {
            return Ok(error_response(error, false, &synapse));
        }
        maybe_write_standalone_inference(
            config,
            clickhouse_connection_info,
            postgres_connection_info,
            deferred_tasks,
            false,
            StandaloneInferenceRecord {
                endpoint: RERANK_ENDPOINT,
                variant_name: format!("{provider_name}::{upstream_name}"),
                model_name: upstream_name.clone(),
                model_provider_name: provider_name.clone(),
                provider_type: provider_name.clone(),
                input: StandaloneInput::Rerank { query, documents },
                output_text: rerank_output_payload(&body),
                raw_request,
                raw_response: serde_json::to_string(&body).unwrap_or_else(|_| body.to_string()),
                usage,
                latency,
                cached: false,
                extra_internal_tags: synapse.observability_tags(&headers),
                tags,
                episode_id,
            },
        )
        .await;
    }

    let mut response = (status, Json(body)).into_response();
    if let Ok(value) = HeaderValue::from_str("application/json") {
        response
            .headers_mut()
            .insert(axum::http::header::CONTENT_TYPE, value);
    }
    synapse.apply_to_response(&mut response);
    Ok(response)
}

/// Ordered `provider::model` candidates for this request (Synapse semantics,
/// mirroring the chat/embedding alias paths):
///
/// - A provider header (or `provider::model` in the body) pins a pair; when a
///   rerank alias lists that pair as a target, the alias's full target list is
///   borrowed with the pinned pair rotated to the head. Otherwise the explicit
///   shorthand is the only candidate.
/// - A bare name resolves through a `[model_aliases]` entry with
///   `task = "rerank"`; its ordered `targets` form the fallback chain, so the
///   chain may cross providers AND models (e.g. `qwen3.7-text-rerank` on
///   alibaba falling back to `qwen3-rerank`).
/// - `fallback_disabled` (`x-synapse-fallback: false`) keeps the head only.
///
/// Candidates whose provider has no rerank upstream configured are dropped
/// (aliases are shared across tasks, so a chain may list chat-only providers).
fn resolve_rerank_candidates(
    model: &str,
    provider: Option<&str>,
    aliases: &ModelAliasTable,
    fallback_disabled: bool,
) -> Result<Vec<String>, Error> {
    let resolved = resolve_openai_compatible_model(model, provider)?;
    let mut candidates: Vec<String> =
        if let Some((provider_type, model_name)) = resolved.split_once("::") {
            match aliases.find_containing(provider_type, model_name, Some("rerank")) {
                Some(alias) => rotated_alias_targets(alias, provider_type, model_name),
                None => vec![resolved],
            }
        } else if let Some(alias) = aliases.resolve(resolved.trim(), Some("rerank")) {
            alias
                .targets
                .iter()
                .map(|target| format!("{}::{}", target.provider_type, target.model_name))
                .collect()
        } else {
            return Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                message: format!(
                    "Rerank model `{resolved}` is not a provider shorthand or rerank model alias. \
                 Use `alibaba::qwen3-rerank`, set `x-synapse-provider`, or add a \
                 `[model_aliases]` entry with `task = \"rerank\"`."
                ),
            }));
        };
    if candidates.len() > 1 && fallback_disabled {
        candidates.truncate(1);
    }
    Ok(candidates
        .into_iter()
        .filter(|candidate| {
            let supported = split_provider_model(candidate)
                .map(|(provider, _)| rerank_provider_supported(provider))
                .unwrap_or(false);
            if !supported {
                tracing::warn!(
                    candidate,
                    "Dropping rerank alias candidate whose provider has no rerank upstream"
                );
            }
            supported
        })
        .collect())
}

/// `provider::model` targets of `alias` with `head_provider::head_model`
/// rotated to the front (no-op when it already leads).
fn rotated_alias_targets(alias: &ModelAlias, head_provider: &str, head_model: &str) -> Vec<String> {
    let head = format!("{head_provider}::{head_model}");
    let mut targets: Vec<String> = alias
        .targets
        .iter()
        .map(|target| format!("{}::{}", target.provider_type, target.model_name))
        .collect();
    if let Some(index) = targets.iter().position(|candidate| candidate == &head)
        && index != 0
    {
        targets.swap(0, index);
    }
    targets
}

/// Providers with a rerank upstream in `rerank_upstream` (dummy is dispatched
/// locally). Aliases are shared across tasks, so chains may list providers
/// that only serve chat/embeddings.
fn rerank_provider_supported(provider: &str) -> bool {
    matches!(provider, "dummy" | "alibaba" | "openrouter" | "siliconflow")
}

/// Outcome of walking a rerank candidate chain.
enum RerankDispatch {
    /// A candidate answered; `index` is its position in the chain (fallback count).
    Served {
        index: usize,
        provider_name: String,
        upstream_name: String,
        raw_request: String,
        status: StatusCode,
        body: Value,
    },
    /// Non-failoverable error (client error) — aborts the chain.
    Error(Error),
    /// Every candidate failed failoverably; carries the last failure to return.
    Exhausted { failure: RerankFailure },
}

/// The last failure of an exhausted chain: either a dispatch error or a
/// non-success upstream response to pass through.
enum RerankFailure {
    Error(Error),
    Upstream { status: StatusCode, body: Value },
}

impl RerankFailure {
    fn into_response(self, synapse: &SynapseRequestContext) -> Response {
        match self {
            RerankFailure::Error(error) => error_response(error, false, synapse),
            RerankFailure::Upstream { status, body } => {
                let mut response = (status, Json(body)).into_response();
                if let Ok(value) = HeaderValue::from_str("application/json") {
                    response
                        .headers_mut()
                        .insert(axum::http::header::CONTENT_TYPE, value);
                }
                synapse.apply_to_response(&mut response);
                response
            }
        }
    }
}

/// Whether an upstream HTTP status should fail over to the next candidate.
/// Same set as Synapse `isFailoverableStatus` / `is_failoverable`:
/// 401/402/403/408/429 and 5xx. Other 4xx are client errors — every
/// provider would reject them, so they short-circuit.
fn is_failoverable_status(status: StatusCode) -> bool {
    matches!(status.as_u16(), 401 | 402 | 403 | 408 | 429 | 500..=599)
}

/// Try each candidate in order. Failoverable failures (network error,
/// timeout, 401/402/403/408/429, 5xx) advance to the next candidate; the
/// first success or non-failoverable failure wins. When the chain is
/// exhausted the last failure is returned. Each attempt gets its own
/// request-timeout budget.
async fn dispatch_with_fallback(
    http_client: &TensorzeroHttpClient,
    candidates: &[String],
    args: &RerankArgs,
    extra: &serde_json::Map<String, Value>,
    request_timeout: Option<Duration>,
) -> RerankDispatch {
    for (index, candidate) in candidates.iter().enumerate() {
        let more_candidates = index + 1 < candidates.len();
        let (provider, upstream_model) = match split_provider_model(candidate) {
            Ok(parts) => parts,
            Err(error) => return RerankDispatch::Error(error),
        };
        let provider_name = provider.to_string();
        let upstream_name = upstream_model.to_string();
        let raw_request = serde_json::to_string(&build_upstream_body(upstream_model, args, extra))
            .unwrap_or_else(|_| "{}".to_string());
        let dispatch_result = Box::pin(run_with_request_timeout(
            request_timeout,
            dispatch_rerank(http_client, provider, upstream_model, args, extra),
        ))
        .await;
        let (status, body) = match dispatch_result {
            Ok(result) => result,
            Err(error) => {
                if !crate::routing::is_failoverable(&error) {
                    return RerankDispatch::Error(error);
                }
                if !more_candidates {
                    return RerankDispatch::Exhausted {
                        failure: RerankFailure::Error(error),
                    };
                }
                tracing::warn!(
                    candidate,
                    error = %error,
                    "Rerank candidate failed; failing over to next alias target"
                );
                continue;
            }
        };
        if !status.is_success() && is_failoverable_status(status) {
            if !more_candidates {
                return RerankDispatch::Exhausted {
                    failure: RerankFailure::Upstream { status, body },
                };
            }
            tracing::warn!(
                candidate,
                status = status.as_u16(),
                "Rerank candidate returned failoverable status; failing over to next alias target"
            );
            continue;
        }
        return RerankDispatch::Served {
            index,
            provider_name,
            upstream_name,
            raw_request,
            status,
            body,
        };
    }
    RerankDispatch::Exhausted {
        failure: RerankFailure::Error(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
            message: "Rerank request had no dispatchable candidates".to_string(),
        })),
    }
}

fn extract_rerank_args(params: &OpenAICompatibleRerankParams) -> Result<RerankArgs, Error> {
    let query = params
        .query
        .clone()
        .or_else(|| params.input.as_ref().and_then(|input| input.query.clone()))
        .ok_or_else(|| {
            Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                message: "`query` is required (or `input.query` for DashScope format)".to_string(),
            })
        })?;
    let documents = params
        .documents
        .clone()
        .or_else(|| {
            params
                .input
                .as_ref()
                .and_then(|input| input.documents.clone())
        })
        .ok_or_else(|| {
            Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                message: "`documents` is required (or `input.documents` for DashScope format)"
                    .to_string(),
            })
        })?;
    if documents.is_empty() {
        return Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
            message: "`documents` must not be empty".to_string(),
        }));
    }
    let top_n = params
        .top_n
        .or_else(|| params.parameters.as_ref().and_then(|p| p.top_n));
    let return_documents = params
        .parameters
        .as_ref()
        .and_then(|p| p.return_documents)
        .or_else(|| params.extra.get("return_documents")?.as_bool());
    // `parameters.*` wins over flat; `instruct` wins over the `instruction` alias.
    let extra_str = |key: &str| {
        params
            .extra
            .get(key)
            .and_then(|value| value.as_str().map(str::to_string))
    };
    let instruct = params
        .parameters
        .as_ref()
        .and_then(|p| p.instruct.as_ref().or(p.instruction.as_ref()).cloned())
        .or_else(|| extra_str("instruct"))
        .or_else(|| extra_str("instruction"));
    Ok(RerankArgs {
        query,
        documents,
        top_n,
        return_documents,
        instruct,
    })
}

async fn dispatch_rerank(
    http_client: &TensorzeroHttpClient,
    provider: &str,
    upstream_model: &str,
    args: &RerankArgs,
    extra: &serde_json::Map<String, Value>,
) -> Result<(StatusCode, Value), Error> {
    if provider == "dummy" {
        return dummy_rerank(upstream_model, &args.documents, args.top_n);
    }

    let (url, api_key_env) = rerank_upstream(provider)?;
    let api_key = std::env::var(api_key_env).map_err(|_| {
        Error::new(ErrorDetails::ApiKeyMissing {
            provider_name: provider.to_string(),
            message: format!("{api_key_env} is not set"),
        })
    })?;

    let body = build_upstream_body(upstream_model, args, extra);

    let request = http_client
        .post(url)
        .bearer_auth(api_key)
        .header("content-type", "application/json")
        .json(&body);
    let response = request.send().await.map_err(|e| {
        Error::new(ErrorDetails::InferenceClient {
            message: format!("Error sending rerank request: {e}"),
            status_code: None,
            provider_type: provider.to_string(),
            api_type: crate::inference::types::ApiType::ChatCompletions,
            raw_request: None,
            raw_response: None,
        })
    })?;

    let status = response.status();
    let bytes = response.bytes().await.map_err(|e| {
        Error::new(ErrorDetails::InferenceServer {
            message: format!("Error reading rerank response: {e}"),
            raw_request: None,
            raw_response: None,
            provider_type: provider.to_string(),
            api_type: crate::inference::types::ApiType::ChatCompletions,
        })
    })?;
    let mut json: Value = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| json!({ "error": String::from_utf8_lossy(&bytes) }));
    unwrap_dashscope_results(&mut json);

    Ok((
        StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
        json,
    ))
}

/// One flat body shape for every provider: `{model, query, documents, top_n?}`
/// plus `return_documents` / `instruct` when resolved. Upstreams without support
/// ignore unknown keys. Consumed keys (`return_documents`, `instruct`, and the
/// `instruction` alias) are skipped in the `extra` passthrough so they never
/// appear twice; `instruction` is never sent upstream (DashScope ignores it).
fn build_upstream_body(
    upstream_model: &str,
    args: &RerankArgs,
    extra: &serde_json::Map<String, Value>,
) -> Value {
    let mut body = serde_json::Map::new();
    body.insert("model".to_string(), json!(upstream_model));
    body.insert("query".to_string(), json!(args.query));
    body.insert("documents".to_string(), json!(args.documents));
    if let Some(top_n) = args.top_n {
        body.insert("top_n".to_string(), json!(top_n));
    }
    if let Some(return_documents) = args.return_documents {
        body.insert("return_documents".to_string(), json!(return_documents));
    }
    if let Some(instruct) = &args.instruct {
        body.insert("instruct".to_string(), json!(instruct));
    }
    for (key, value) in extra {
        if matches!(
            key.as_str(),
            "model"
                | "query"
                | "documents"
                | "top_n"
                | "input"
                | "parameters"
                | "return_documents"
                | "instruct"
                | "instruction"
        ) {
            continue;
        }
        body.insert(key.clone(), value.clone());
    }
    Value::Object(body)
}

fn split_provider_model(model: &str) -> Result<(&str, &str), Error> {
    model.split_once("::").ok_or_else(|| {
        Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
            message: format!(
                "Rerank model `{model}` is not a provider shorthand. Use `alibaba::qwen3-rerank` or set `x-synapse-provider`."
            ),
        })
    })
}

fn rerank_upstream(provider: &str) -> Result<(Url, &'static str), Error> {
    match provider {
        "alibaba" => {
            let base = openai_compatible_shorthand_api_base(
                "ALIBABA_RERANK_BASE_URL",
                ALIBABA_RERANK_DEFAULT_API_ROOT,
                true,
            )?;
            Ok((join_path(&base, "reranks")?, "ALIBABA_API_KEY"))
        }
        "openrouter" => {
            let base = openai_compatible_shorthand_api_base(
                "OPENROUTER_BASE_URL",
                "https://openrouter.ai/api",
                true,
            )?;
            Ok((join_path(&base, "rerank")?, "OPENROUTER_API_KEY"))
        }
        "siliconflow" => {
            let base = openai_compatible_shorthand_api_base(
                "SILICONFLOW_BASE_URL",
                SILICONFLOW_DEFAULT_API_ROOT,
                true,
            )?;
            Ok((join_path(&base, "rerank")?, "SILICONFLOW_API_KEY"))
        }
        other => Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
            message: format!("Rerank is not configured for provider `{other}`"),
        })),
    }
}

fn join_path(base: &Url, segment: &str) -> Result<Url, Error> {
    let mut url = base.clone();
    if !url.path().ends_with('/') {
        url.set_path(&format!("{}/", url.path()));
    }
    url.join(segment).map_err(|e| {
        Error::new(ErrorDetails::InvalidBaseUrl {
            message: e.to_string(),
        })
    })
}

fn unwrap_dashscope_results(json: &mut Value) {
    if json.get("results").is_some() {
        return;
    }
    let Some(nested) = json
        .get("output")
        .and_then(|output| output.get("results"))
        .cloned()
    else {
        return;
    };
    let Some(obj) = json.as_object_mut() else {
        return;
    };
    obj.insert("results".to_string(), nested);
}

fn apply_rerank_cost(
    config: &crate::config::Config,
    provider: &str,
    model: &str,
    body: &Value,
) -> Usage {
    let mut usage = usage_from_json(body);
    let Some(cost_config) = config.rerank_models.cost(model, provider) else {
        return usage;
    };
    let mut billed = body.clone();
    if let Some(obj) = billed.as_object_mut() {
        let mut extra = obj
            .remove("_tensorzero")
            .and_then(|value| value.as_object().cloned())
            .unwrap_or_default();
        extra.insert("searches".to_string(), json!(1));
        obj.insert("_tensorzero".to_string(), Value::Object(extra));
    }
    apply_computed_cost(
        &mut usage,
        &billed.to_string(),
        cost_config,
        ResponseMode::NonStreaming,
    );
    usage
}

fn overlay_rerank_usage(body: &mut Value, usage: &Usage) {
    let Some(obj) = body.as_object_mut() else {
        return;
    };
    let usage_value = obj.entry("usage").or_insert_with(|| json!({}));
    let Some(map) = usage_value.as_object_mut() else {
        return;
    };
    if let Some(tokens) = usage.input_tokens {
        map.entry("prompt_tokens").or_insert(json!(tokens));
        map.entry("total_tokens").or_insert(json!(tokens));
    }
    if let Some(cost) = usage.cost {
        map.insert("tensorzero_cost".to_string(), json!(decimal_as_f64(cost)));
        let currency = usage.currency.unwrap_or(tensorzero_types::Currency::USD);
        if usage.currency.is_some() {
            map.insert("tensorzero_currency".to_string(), json!(currency.as_str()));
        }
        map.insert(
            "tensorzero_costs".to_string(),
            json!({ currency.as_str(): decimal_as_f64(cost) }),
        );
    }
}

fn decimal_as_f64(value: rust_decimal::Decimal) -> f64 {
    use rust_decimal::prelude::ToPrimitive;
    value.to_f64().unwrap_or(0.0)
}

fn dummy_rerank(
    model: &str,
    documents: &[String],
    top_n: Option<u32>,
) -> Result<(StatusCode, Value), Error> {
    if model.starts_with("error") {
        return Err(Error::new(ErrorDetails::InferenceClient {
            message: format!("Error sending request to Dummy provider for model '{model}'."),
            status_code: Some(StatusCode::INTERNAL_SERVER_ERROR),
            provider_type: "dummy".to_string(),
            api_type: crate::inference::types::ApiType::ChatCompletions,
            raw_request: None,
            raw_response: None,
        }));
    }
    let take = top_n
        .and_then(|n| usize::try_from(n).ok())
        .unwrap_or(documents.len())
        .min(documents.len());
    let results: Vec<CohereRerankResult> = documents
        .iter()
        .take(take)
        .enumerate()
        .map(|(index, text)| CohereRerankResult {
            index,
            relevance_score: 1.0 - f64::from(u32::try_from(index).unwrap_or(u32::MAX)) * 0.01,
            document: Some(CohereRerankDocument { text: text.clone() }),
        })
        .collect();
    Ok((
        StatusCode::OK,
        json!({
            "results": results,
            "usage": { "total_tokens": 0 }
        }),
    ))
}

#[cfg(test)]
mod tests {
    use googletest::prelude::*;
    use googletest_matchers::matches_json_literal;

    use super::*;
    use crate::model_alias::ModelAliasTarget;
    use std::sync::Arc;

    #[test]
    fn extract_cohere_style() {
        let params: OpenAICompatibleRerankParams = serde_json::from_value(json!({
            "model": "qwen3-rerank",
            "query": "capital",
            "documents": ["Paris", "London"],
            "top_n": 1
        }))
        .unwrap();
        let args = extract_rerank_args(&params).unwrap();
        assert_eq!(args.query, "capital");
        assert_eq!(args.documents, vec!["Paris", "London"]);
        assert_eq!(args.top_n, Some(1));
        assert_eq!(args.return_documents, None);
        assert_eq!(args.instruct, None);
    }

    #[test]
    fn extract_dashscope_style() {
        let params: OpenAICompatibleRerankParams = serde_json::from_value(json!({
            "model": "qwen3-rerank",
            "input": { "query": "capital", "documents": ["Paris"] },
            "parameters": { "top_n": 2 }
        }))
        .unwrap();
        let args = extract_rerank_args(&params).unwrap();
        assert_eq!(args.query, "capital");
        assert_eq!(args.documents, vec!["Paris"]);
        assert_eq!(args.top_n, Some(2));
    }

    #[gtest]
    fn extract_dashscope_parameters_instruct_and_return_documents() {
        let params: OpenAICompatibleRerankParams = serde_json::from_value(json!({
            "model": "qwen3.7-text-rerank",
            "input": { "query": "capital", "documents": ["Paris"] },
            "parameters": { "instruct": "rank by relevance", "return_documents": false }
        }))
        .expect("DashScope-style params should parse");
        let args = extract_rerank_args(&params).expect("extract_rerank_args should succeed");
        expect_that!(args.instruct, some(eq("rank by relevance")));
        expect_that!(args.return_documents, some(eq(false)));
    }

    #[gtest]
    fn extract_flat_instruct_and_return_documents_fallback() {
        let params: OpenAICompatibleRerankParams = serde_json::from_value(json!({
            "model": "qwen3.7-text-rerank",
            "query": "capital",
            "documents": ["Paris"],
            "instruct": "flat instruct",
            "return_documents": true
        }))
        .expect("flat params should parse");
        let args = extract_rerank_args(&params).expect("extract_rerank_args should succeed");
        expect_that!(args.instruct, some(eq("flat instruct")));
        expect_that!(args.return_documents, some(eq(true)));
    }

    #[gtest]
    fn extract_nested_instruction_normalizes_to_instruct() {
        let params: OpenAICompatibleRerankParams = serde_json::from_value(json!({
            "model": "qwen3.7-text-rerank",
            "input": { "query": "capital", "documents": ["Paris"] },
            "parameters": { "instruction": "nested instruction" }
        }))
        .expect("params should parse");
        let args = extract_rerank_args(&params).expect("extract_rerank_args should succeed");
        expect_that!(args.instruct, some(eq("nested instruction")));
    }

    #[gtest]
    fn extract_flat_instruction_normalizes_to_instruct() {
        let params: OpenAICompatibleRerankParams = serde_json::from_value(json!({
            "model": "qwen3.7-text-rerank",
            "query": "capital",
            "documents": ["Paris"],
            "instruction": "flat instruction"
        }))
        .expect("params should parse");
        let args = extract_rerank_args(&params).expect("extract_rerank_args should succeed");
        expect_that!(args.instruct, some(eq("flat instruction")));
    }

    #[gtest]
    fn extract_parameters_win_over_flat() {
        let params: OpenAICompatibleRerankParams = serde_json::from_value(json!({
            "model": "qwen3.7-text-rerank",
            "input": { "query": "capital", "documents": ["Paris"] },
            "parameters": { "instruct": "nested instruct", "return_documents": false },
            "instruct": "flat instruct",
            "instruction": "flat instruction",
            "return_documents": true
        }))
        .expect("params should parse");
        let args = extract_rerank_args(&params).expect("extract_rerank_args should succeed");
        expect_that!(args.instruct, some(eq("nested instruct")));
        expect_that!(args.return_documents, some(eq(false)));

        // `parameters.instruction` still beats flat `instruct`.
        let params: OpenAICompatibleRerankParams = serde_json::from_value(json!({
            "model": "qwen3.7-text-rerank",
            "input": { "query": "capital", "documents": ["Paris"] },
            "parameters": { "instruction": "nested instruction" },
            "instruct": "flat instruct"
        }))
        .expect("params should parse");
        let args = extract_rerank_args(&params).expect("extract_rerank_args should succeed");
        expect_that!(args.instruct, some(eq("nested instruction")));
    }

    #[test]
    fn unwrap_nested_results() {
        let mut json = json!({ "output": { "results": [{ "index": 0, "relevance_score": 1.0 }] } });
        unwrap_dashscope_results(&mut json);
        assert!(json.get("results").is_some());
    }

    #[test]
    fn dummy_scores_preserve_index() {
        let (status, body) = dummy_rerank("good", &["a".into(), "b".into()], Some(1)).unwrap();
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["results"].as_array().unwrap().len(), 1);
        assert_eq!(body["results"][0]["index"], 0);
    }

    #[test]
    fn build_upstream_body_rewrites_model_and_keeps_extra() {
        let mut extra = serde_json::Map::new();
        extra.insert("custom_key".to_string(), json!("kept"));
        extra.insert("query".to_string(), json!("should-not-win"));
        let args = RerankArgs {
            query: "capital".to_string(),
            documents: vec!["Paris".to_string()],
            top_n: Some(1),
            return_documents: Some(true),
            instruct: None,
        };
        let body = build_upstream_body("qwen3-rerank", &args, &extra);
        assert_eq!(body["model"], "qwen3-rerank");
        assert_eq!(body["query"], "capital");
        assert_eq!(body["top_n"], 1);
        assert_eq!(body["return_documents"], true);
        assert_eq!(body["custom_key"], "kept");
    }

    #[gtest]
    fn build_body_forwards_instruct_and_return_documents_flat() {
        let args = RerankArgs {
            query: "capital".to_string(),
            documents: vec!["Paris".to_string(), "London".to_string()],
            top_n: Some(1),
            return_documents: Some(true),
            instruct: Some("rank by relevance".to_string()),
        };
        // Consumed keys in `extra` must be skipped so they never appear twice;
        // `instruction` must never be sent upstream.
        let mut extra = serde_json::Map::new();
        extra.insert("instruct".to_string(), json!("stale flat instruct"));
        extra.insert("instruction".to_string(), json!("stale flat instruction"));
        extra.insert("return_documents".to_string(), json!(false));
        let body = build_upstream_body("qwen3.7-text-rerank", &args, &extra);
        expect_that!(
            body,
            matches_json_literal!({
                "model": "qwen3.7-text-rerank",
                "query": "capital",
                "documents": ["Paris", "London"],
                "top_n": 1,
                "return_documents": true,
                "instruct": "rank by relevance"
            })
        );
    }

    #[gtest]
    fn resolve_bare_name_returns_full_alias_chain_dropping_unsupported_providers() {
        let aliases = ModelAliasTable {
            aliases: vec![ModelAlias {
                name: Arc::from("qwen3.7-text-rerank"),
                task: Some(Arc::from("rerank")),
                targets: vec![
                    ModelAliasTarget {
                        provider_type: Arc::from("alibaba"),
                        model_name: Arc::from("qwen3.7-text-rerank"),
                    },
                    ModelAliasTarget {
                        provider_type: Arc::from("alibaba"),
                        model_name: Arc::from("qwen3-rerank"),
                    },
                    // Chat-only provider on a shared alias — must be dropped.
                    ModelAliasTarget {
                        provider_type: Arc::from("deepseek"),
                        model_name: Arc::from("deepseek-v4-flash"),
                    },
                ],
                min_tokens_per_sec: None,
            }],
        };
        expect_eq!(
            resolve_rerank_candidates("qwen3.7-text-rerank", None, &aliases, false).unwrap(),
            vec![
                "alibaba::qwen3.7-text-rerank".to_string(),
                "alibaba::qwen3-rerank".to_string(),
            ]
        );
    }

    #[gtest]
    fn resolve_fallback_disabled_keeps_head_only() {
        let aliases = ModelAliasTable {
            aliases: vec![ModelAlias {
                name: Arc::from("qwen3.7-text-rerank"),
                task: Some(Arc::from("rerank")),
                targets: vec![
                    ModelAliasTarget {
                        provider_type: Arc::from("alibaba"),
                        model_name: Arc::from("qwen3.7-text-rerank"),
                    },
                    ModelAliasTarget {
                        provider_type: Arc::from("alibaba"),
                        model_name: Arc::from("qwen3-rerank"),
                    },
                ],
                min_tokens_per_sec: None,
            }],
        };
        expect_eq!(
            resolve_rerank_candidates("qwen3.7-text-rerank", None, &aliases, true).unwrap(),
            vec!["alibaba::qwen3.7-text-rerank".to_string()]
        );
    }

    #[gtest]
    fn resolve_provider_header_without_alias_keeps_single_candidate() {
        let aliases = ModelAliasTable::default();
        expect_eq!(
            resolve_rerank_candidates("qwen3-rerank", Some("dummy"), &aliases, false).unwrap(),
            vec!["dummy::qwen3-rerank".to_string()]
        );
    }

    #[gtest]
    fn resolve_explicit_shorthand_borrows_alias_chain_rotated_to_head() {
        let aliases = ModelAliasTable {
            aliases: vec![ModelAlias {
                name: Arc::from("qwen3.7-text-rerank"),
                task: Some(Arc::from("rerank")),
                targets: vec![
                    ModelAliasTarget {
                        provider_type: Arc::from("alibaba"),
                        model_name: Arc::from("qwen3.7-text-rerank"),
                    },
                    ModelAliasTarget {
                        provider_type: Arc::from("alibaba"),
                        model_name: Arc::from("qwen3-rerank"),
                    },
                ],
                min_tokens_per_sec: None,
            }],
        };
        // Pinning the tail target rotates it to the head but keeps the chain
        // (chat/embedding Synapse semantics).
        expect_eq!(
            resolve_rerank_candidates("alibaba::qwen3-rerank", None, &aliases, false).unwrap(),
            vec![
                "alibaba::qwen3-rerank".to_string(),
                "alibaba::qwen3.7-text-rerank".to_string(),
            ]
        );
        // A provider header pin behaves the same as an explicit shorthand.
        expect_eq!(
            resolve_rerank_candidates("qwen3-rerank", Some("alibaba"), &aliases, false).unwrap(),
            vec![
                "alibaba::qwen3-rerank".to_string(),
                "alibaba::qwen3.7-text-rerank".to_string(),
            ]
        );
    }

    #[gtest]
    fn resolve_unknown_bare_name_errors() {
        let aliases = ModelAliasTable::default();
        expect_that!(
            resolve_rerank_candidates("mystery-rerank", None, &aliases, false).is_err(),
            eq(true)
        );
    }

    #[gtest]
    fn failoverable_status_matches_synapse_set() {
        expect_that!(is_failoverable_status(StatusCode::UNAUTHORIZED), eq(true));
        expect_that!(
            is_failoverable_status(StatusCode::PAYMENT_REQUIRED),
            eq(true)
        );
        expect_that!(is_failoverable_status(StatusCode::FORBIDDEN), eq(true));
        expect_that!(
            is_failoverable_status(StatusCode::REQUEST_TIMEOUT),
            eq(true)
        );
        expect_that!(
            is_failoverable_status(StatusCode::TOO_MANY_REQUESTS),
            eq(true)
        );
        expect_that!(is_failoverable_status(StatusCode::BAD_GATEWAY), eq(true));
        // Client errors short-circuit: every provider would reject them.
        expect_that!(is_failoverable_status(StatusCode::BAD_REQUEST), eq(false));
        expect_that!(is_failoverable_status(StatusCode::NOT_FOUND), eq(false));
        expect_that!(
            is_failoverable_status(StatusCode::UNPROCESSABLE_ENTITY),
            eq(false)
        );
    }

    #[gtest]
    #[tokio::test]
    async fn dispatch_falls_over_on_failoverable_error() {
        // `dummy::error` returns a 500 InferenceClient error — failoverable.
        let args = RerankArgs {
            query: "capital".to_string(),
            documents: vec!["Paris".to_string()],
            top_n: None,
            return_documents: None,
            instruct: None,
        };
        let dispatch = dispatch_with_fallback(
            &TensorzeroHttpClient::new_testing().expect("test http client"),
            &["dummy::error".to_string(), "dummy::good".to_string()],
            &args,
            &serde_json::Map::new(),
            None,
        )
        .await;
        let RerankDispatch::Served {
            index,
            provider_name,
            upstream_name,
            status,
            ..
        } = dispatch
        else {
            panic!("expected RerankDispatch::Served");
        };
        expect_that!(index, eq(1));
        expect_eq!(provider_name, "dummy".to_string());
        expect_eq!(upstream_name, "good".to_string());
        expect_that!(status, eq(StatusCode::OK));
    }

    #[gtest]
    #[tokio::test]
    async fn dispatch_exhausted_returns_last_error() {
        let args = RerankArgs {
            query: "capital".to_string(),
            documents: vec!["Paris".to_string()],
            top_n: None,
            return_documents: None,
            instruct: None,
        };
        let dispatch = dispatch_with_fallback(
            &TensorzeroHttpClient::new_testing().expect("test http client"),
            &["dummy::error".to_string()],
            &args,
            &serde_json::Map::new(),
            None,
        )
        .await;
        let RerankDispatch::Exhausted { .. } = dispatch else {
            panic!("expected RerankDispatch::Exhausted");
        };
    }

    #[test]
    fn alibaba_rerank_url_uses_compatible_api_reranks() {
        let (url, key) = rerank_upstream("alibaba").unwrap();
        assert_eq!(key, "ALIBABA_API_KEY");
        assert!(
            url.path().ends_with("/reranks"),
            "unexpected rerank url {url}"
        );
    }

    #[test]
    fn apply_alibaba_rerank_cost_records_cny() {
        use crate::config::rerank::{RerankModelTable, UninitializedRerankModelConfig};
        use rust_decimal::Decimal;
        use std::sync::Arc;
        use tensorzero_types::Currency;

        let models: HashMap<Arc<str>, UninitializedRerankModelConfig> = toml::from_str(
            r#"
[qwen3-rerank.providers.alibaba]
currency = "CNY"
cost = [
  { pointer = "/usage/total_tokens", cost_per_million = 0.5, usage = "input" },
]
"#,
        )
        .expect("rerank cost toml");
        let config = crate::config::Config {
            rerank_models: RerankModelTable::load(models).expect("load rerank cost"),
            ..Default::default()
        };
        let body = json!({ "usage": { "total_tokens": 1_000_000 } });
        let usage = apply_rerank_cost(&config, "alibaba", "qwen3-rerank", &body);
        assert_eq!(usage.input_tokens, Some(1_000_000));
        assert_eq!(usage.cost, Some(Decimal::new(5, 1)));
        assert_eq!(usage.currency, Some(Currency::CNY));

        let mut overlaid = body;
        overlay_rerank_usage(&mut overlaid, &usage);
        assert_eq!(overlaid["usage"]["tensorzero_currency"], "CNY");
        assert_eq!(overlaid["usage"]["tensorzero_costs"]["CNY"], 0.5);
    }

    #[test]
    fn apply_openrouter_rerank_cost_falls_back_to_search_rate() {
        use crate::config::rerank::{RerankModelTable, UninitializedRerankModelConfig};
        use rust_decimal::Decimal;
        use std::sync::Arc;
        use tensorzero_types::Currency;

        let models: HashMap<Arc<str>, UninitializedRerankModelConfig> = toml::from_str(
            r#"
["cohere/rerank-v3.5".providers.openrouter]
currency = "USD"
cost = [
  { pointer = "/usage/cost", cost_per_unit = 1 },
  { pointer = "/_tensorzero/searches", cost_per_unit = 0.002, skip_if_pointer = "/usage/cost" },
]
"#,
        )
        .expect("rerank cost toml");
        let config = crate::config::Config {
            rerank_models: RerankModelTable::load(models).expect("load rerank cost"),
            ..Default::default()
        };
        let usage = apply_rerank_cost(
            &config,
            "openrouter",
            "cohere/rerank-v3.5",
            &json!({ "results": [] }),
        );
        assert_eq!(usage.cost, Some(Decimal::new(2, 3)));
        assert_eq!(usage.currency, Some(Currency::USD));
    }
}

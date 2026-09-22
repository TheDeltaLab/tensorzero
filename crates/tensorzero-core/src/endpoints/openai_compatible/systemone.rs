// Modified by Delta-AI under Apache 2.0
//! `POST /v1/systemone` for TypeSafe System One models such as Jev.
//!
//! Jev does not generate text. Callers send `{ model, state, questions }` and
//! receive typed answers. TensorZero forwards the call to OpenRouter's System
//! One API (`POST /api/v1/systemone`) with `OPENROUTER_API_KEY`.
//!
//! Bare names `jev` and `jev-latest` route to `~typesafe/jev-latest`.
//! `jev-1.13` and `jev-1.13.0` route to `typesafe/jev-1.13`. An explicit
//! `openrouter::…` shorthand is forwarded as-is after the same normalization,
//! so a future Jev id works without a gateway change.

use std::collections::HashMap;
use std::time::Instant;

use axum::Json;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use rust_decimal::Decimal;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use url::Url;

use crate::endpoints::standalone_inference::{
    SYSTEMONE_ENDPOINT, StandaloneInferenceRecord, StandaloneInput,
    maybe_write_standalone_inference, systemone_output_payload,
};
use crate::error::{Error, ErrorDetails};
use crate::http::TensorzeroHttpClient;
use crate::inference::types::{Latency, Usage};
use crate::model::openai_compatible_shorthand_api_base;
use crate::model_alias::ModelAliasTable;
use crate::utils::gateway::{AppState, AppStateData};

use super::OpenAIStructuredJson;
use super::infer::error_response;
use super::synapse::{
    SynapseRequestContext, overlay_compat_headers, resolve_openai_compatible_model,
    run_with_request_timeout, served_by_from_model_name,
};

const TENSORZERO_MODEL_NAME_PREFIX: &str = "tensorzero::model_name::";
/// $0.042 per million input tokens. Output tokens are free.
const JEV_INPUT_USD_PER_TOKEN: Decimal = Decimal::from_parts(42, 0, 0, false, 9);

#[derive(Debug, Deserialize)]
pub struct SystemOneParams {
    pub model: String,
    pub state: Value,
    pub questions: Map<String, Value>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

pub async fn systemone_handler(
    State(AppStateData {
        http_client,
        config,
        clickhouse_connection_info,
        postgres_connection_info,
        deferred_tasks,
        ..
    }): AppState,
    headers: HeaderMap,
    OpenAIStructuredJson(params): OpenAIStructuredJson<SystemOneParams>,
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
    let model = match resolve_systemone_model(
        &params.model,
        synapse.provider.as_deref(),
        &config.models.model_aliases,
    ) {
        Ok(model) => model,
        Err(error) => return Ok(error_response(error, false, &synapse)),
    };
    if let Err(error) = validate_systemone_body(&params.state, &params.questions) {
        return Ok(error_response(error, false, &synapse));
    }
    synapse.served_by = Some(served_by_from_model_name(&model));

    let (provider, upstream_model) = match split_provider_model(&model) {
        Ok(parts) => parts,
        Err(error) => return Ok(error_response(error, false, &synapse)),
    };
    let provider_name = provider.to_string();
    let upstream_name = upstream_model.to_string();
    let upstream_body = build_upstream_body(
        upstream_model,
        &params.state,
        &params.questions,
        &params.extra,
    );
    let raw_request = serde_json::to_string(&upstream_body).unwrap_or_else(|_| "{}".to_string());
    let start = Instant::now();
    let dispatch_result = Box::pin(run_with_request_timeout(
        synapse.request_timeout,
        dispatch_systemone(&http_client, provider, upstream_body),
    ))
    .await;
    let latency = Latency::NonStreaming {
        response_time: start.elapsed(),
    };

    let (status, mut body) = match dispatch_result {
        Ok(result) => result,
        Err(error) => return Ok(error_response(error, false, &synapse)),
    };

    if status.is_success() {
        let usage = usage_from_systemone(&body);
        overlay_systemone_usage(&mut body, &usage);
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
                endpoint: SYSTEMONE_ENDPOINT,
                variant_name: model,
                model_name: upstream_name,
                model_provider_name: provider_name.clone(),
                provider_type: provider_name,
                input: StandaloneInput::SystemOne {
                    state: state_text(&params.state),
                    questions: serde_json::to_string(&params.questions)
                        .unwrap_or_else(|_| "{}".to_string()),
                },
                output_text: systemone_output_payload(&body),
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

/// Provider header wins. Otherwise a `[model_aliases]` entry with
/// `task = "systemone"` supplies `provider::model`. Known bare Jev names
/// default to OpenRouter.
pub(crate) fn resolve_systemone_model(
    model: &str,
    provider: Option<&str>,
    aliases: &ModelAliasTable,
) -> Result<String, Error> {
    let trimmed = model.trim();
    let model = trimmed
        .strip_prefix(TENSORZERO_MODEL_NAME_PREFIX)
        .unwrap_or(trimmed);
    let resolved = resolve_openai_compatible_model(model, provider)?;
    if let Some((provider_name, upstream)) = resolved.split_once("::") {
        return Ok(canonicalize_systemone_target(provider_name, upstream));
    }
    if let Some(alias) = aliases.resolve(model, Some("systemone"))
        && let Some(target) = alias.targets.first()
    {
        return Ok(canonicalize_systemone_target(
            target.provider_type.as_ref(),
            target.model_name.as_ref(),
        ));
    }
    if let Some(upstream) = builtin_jev_model(model) {
        return Ok(format!("openrouter::{upstream}"));
    }
    Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
        message: format!(
            "System One model `{model}` is not supported. Use `jev`, `jev-latest`, `jev-1.13`, or `openrouter::typesafe/<model>`."
        ),
    }))
}

fn canonicalize_systemone_target(provider: &str, model: &str) -> String {
    let model = if provider == "openrouter" {
        normalize_openrouter_systemone_model(model)
    } else {
        model.to_string()
    };
    format!("{provider}::{model}")
}

fn builtin_jev_model(model: &str) -> Option<String> {
    match model {
        "jev" | "jev-latest" | "typesafe/jev" | "typesafe/jev-latest" | "~typesafe/jev-latest" => {
            Some("~typesafe/jev-latest".to_string())
        }
        "jev-1.13" | "jev-1.13.0" | "typesafe/jev-1.13" | "typesafe/jev-1.13.0" => {
            Some("typesafe/jev-1.13".to_string())
        }
        "jev-preview" | "typesafe/jev-preview" | "~typesafe/jev-preview" => {
            Some("~typesafe/jev-preview".to_string())
        }
        _ => None,
    }
}

fn normalize_openrouter_systemone_model(model: &str) -> String {
    builtin_jev_model(model).unwrap_or_else(|| model.to_string())
}

fn validate_systemone_body(state: &Value, questions: &Map<String, Value>) -> Result<(), Error> {
    match state {
        Value::String(text) if text.is_empty() => {
            return Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                message: "`state` must not be empty".to_string(),
            }));
        }
        Value::Array(items) if items.is_empty() => {
            return Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                message: "`state` must not be an empty array".to_string(),
            }));
        }
        Value::String(_) | Value::Object(_) | Value::Array(_) => {}
        _ => {
            return Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                message: "`state` must be a string, object, or array of text".to_string(),
            }));
        }
    }
    if questions.is_empty() {
        return Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
            message: "`questions` must contain at least one question".to_string(),
        }));
    }
    Ok(())
}

async fn dispatch_systemone(
    http_client: &TensorzeroHttpClient,
    provider: &str,
    body: Value,
) -> Result<(StatusCode, Value), Error> {
    if provider == "dummy" {
        let model = body.get("model").and_then(Value::as_str).unwrap_or("good");
        let questions = body
            .get("questions")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        return dummy_systemone(model, &questions);
    }
    if provider != "openrouter" {
        return Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
            message: format!(
                "System One is only configured for OpenRouter (provider `{provider}`)"
            ),
        }));
    }

    let url = openrouter_systemone_url()?;
    let api_key = std::env::var("OPENROUTER_API_KEY").map_err(|_| {
        Error::new(ErrorDetails::ApiKeyMissing {
            provider_name: "openrouter".to_string(),
            message: "OPENROUTER_API_KEY is not set".to_string(),
        })
    })?;
    let response = http_client
        .post(url)
        .bearer_auth(api_key)
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                message: format!("Error sending System One request: {e}"),
                status_code: None,
                provider_type: "openrouter".to_string(),
                api_type: crate::inference::types::ApiType::ChatCompletions,
                raw_request: None,
                raw_response: None,
            })
        })?;
    let status = response.status();
    let bytes = response.bytes().await.map_err(|e| {
        Error::new(ErrorDetails::InferenceServer {
            message: format!("Error reading System One response: {e}"),
            raw_request: None,
            raw_response: None,
            provider_type: "openrouter".to_string(),
            api_type: crate::inference::types::ApiType::ChatCompletions,
        })
    })?;
    let json: Value = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| json!({ "error": String::from_utf8_lossy(&bytes) }));
    Ok((
        StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
        json,
    ))
}

fn openrouter_systemone_url() -> Result<Url, Error> {
    let base = openai_compatible_shorthand_api_base(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api",
        true,
    )?;
    join_path(&base, "systemone")
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

fn build_upstream_body(
    upstream_model: &str,
    state: &Value,
    questions: &Map<String, Value>,
    extra: &Map<String, Value>,
) -> Value {
    let mut body = Map::new();
    body.insert("model".to_string(), json!(upstream_model));
    body.insert("state".to_string(), state.clone());
    body.insert("questions".to_string(), Value::Object(questions.clone()));
    for (key, value) in extra {
        if matches!(key.as_str(), "model" | "state" | "questions") {
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
                "System One model `{model}` is not a provider shorthand. Use `jev` or `openrouter::typesafe/jev-1.13`."
            ),
        })
    })
}

pub(crate) fn usage_from_systemone(body: &Value) -> Usage {
    let Some(usage) = body.get("usage") else {
        return Usage::default();
    };
    let as_u32 = |key: &str| usage.get(key).and_then(Value::as_u64).map(|n| n as u32);
    let input_tokens = as_u32("input_tokens").or_else(|| as_u32("prompt_tokens"));
    let output_tokens = as_u32("output_tokens").or_else(|| as_u32("completion_tokens"));
    let reported_cost = usage.get("cost").and_then(json_decimal);
    let cost = reported_cost
        .or_else(|| input_tokens.map(|tokens| Decimal::from(tokens) * JEV_INPUT_USD_PER_TOKEN));
    Usage {
        input_tokens,
        output_tokens,
        provider_cache_read_input_tokens: None,
        provider_cache_write_input_tokens: None,
        currency: cost.map(|_| tensorzero_types::Currency::USD),
        cost,
    }
}

fn json_decimal(value: &Value) -> Option<Decimal> {
    match value {
        Value::Number(number) => number.to_string().parse().ok(),
        Value::String(text) => text.parse().ok(),
        _ => None,
    }
}

fn overlay_systemone_usage(body: &mut Value, usage: &Usage) {
    let Some(obj) = body.as_object_mut() else {
        return;
    };
    let usage_value = obj.entry("usage").or_insert_with(|| json!({}));
    let Some(map) = usage_value.as_object_mut() else {
        return;
    };
    if let Some(tokens) = usage.input_tokens {
        map.entry("input_tokens").or_insert(json!(tokens));
    }
    if let Some(tokens) = usage.output_tokens {
        map.entry("output_tokens").or_insert(json!(tokens));
    }
    if let Some(cost) = usage.cost {
        map.insert("tensorzero_cost".to_string(), json!(decimal_as_f64(cost)));
        let currency = usage.currency.unwrap_or(tensorzero_types::Currency::USD);
        map.insert("tensorzero_currency".to_string(), json!(currency.as_str()));
        map.insert(
            "tensorzero_costs".to_string(),
            json!({ currency.as_str(): decimal_as_f64(cost) }),
        );
    }
}

fn decimal_as_f64(value: Decimal) -> f64 {
    use rust_decimal::prelude::ToPrimitive;
    value.to_f64().unwrap_or(0.0)
}

fn state_text(state: &Value) -> String {
    match state {
        Value::String(text) => text.clone(),
        other => serde_json::to_string(other).unwrap_or_else(|_| other.to_string()),
    }
}

fn dummy_systemone(
    model: &str,
    questions: &Map<String, Value>,
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
    let mut answers = Map::new();
    for (id, question) in questions {
        answers.insert(id.clone(), dummy_answer(question));
    }
    Ok((
        StatusCode::OK,
        json!({
            "id": "dummy-systemone",
            "model": model,
            "provider": "dummy",
            "answers": answers,
            "usage": { "input_tokens": 12, "output_tokens": 3 }
        }),
    ))
}

fn dummy_answer(question: &Value) -> Value {
    match question.get("type").and_then(Value::as_str) {
        Some("choice") => {
            let choice = question
                .get("criteria")
                .and_then(Value::as_object)
                .and_then(|criteria| criteria.keys().next().cloned())
                .unwrap_or_else(|| "unknown".to_string());
            json!({
                "type": "choice",
                "choice": choice,
                "probabilities": { choice.clone(): 1.0 },
                "confidence": 1.0
            })
        }
        Some("score") => json!({
            "type": "score",
            "score": 0.0,
            "legend": { "0": "low" },
            "probabilities": { "0": 1.0 },
            "confidence": 1.0
        }),
        _ => json!({ "type": "noul", "noul": 0.5 }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use googletest::prelude::*;
    use std::sync::Arc;

    use crate::model_alias::{ModelAlias, ModelAliasTarget};

    fn aliases_with_jev_override() -> ModelAliasTable {
        ModelAliasTable {
            aliases: vec![ModelAlias {
                name: Arc::from("jev"),
                task: Some(Arc::from("systemone")),
                targets: vec![ModelAliasTarget {
                    provider_type: Arc::from("dummy"),
                    model_name: Arc::from("good"),
                }],
                min_tokens_per_sec: None,
            }],
        }
    }

    #[gtest]
    fn bare_jev_names_route_to_openrouter() {
        let aliases = ModelAliasTable::default();
        expect_eq!(
            resolve_systemone_model("jev", None, &aliases).unwrap(),
            "openrouter::~typesafe/jev-latest"
        );
        expect_eq!(
            resolve_systemone_model("jev-latest", None, &aliases).unwrap(),
            "openrouter::~typesafe/jev-latest"
        );
        expect_eq!(
            resolve_systemone_model("jev-1.13", None, &aliases).unwrap(),
            "openrouter::typesafe/jev-1.13"
        );
        expect_eq!(
            resolve_systemone_model("jev-1.13.0", None, &aliases).unwrap(),
            "openrouter::typesafe/jev-1.13"
        );
        expect_eq!(
            resolve_systemone_model("typesafe/jev-1.13", None, &aliases).unwrap(),
            "openrouter::typesafe/jev-1.13"
        );
        expect_eq!(
            resolve_systemone_model("~typesafe/jev-latest", None, &aliases).unwrap(),
            "openrouter::~typesafe/jev-latest"
        );
        expect_eq!(
            resolve_systemone_model("tensorzero::model_name::jev", None, &aliases).unwrap(),
            "openrouter::~typesafe/jev-latest"
        );
    }

    #[gtest]
    fn explicit_openrouter_shorthand_keeps_unknown_future_ids() {
        let aliases = ModelAliasTable::default();
        expect_eq!(
            resolve_systemone_model("openrouter::typesafe/jev-9", None, &aliases).unwrap(),
            "openrouter::typesafe/jev-9"
        );
        expect_eq!(
            resolve_systemone_model("jev-1.13", Some("openrouter"), &aliases).unwrap(),
            "openrouter::typesafe/jev-1.13"
        );
    }

    #[gtest]
    fn systemone_alias_overrides_builtin_jev() {
        let resolved = resolve_systemone_model("jev", None, &aliases_with_jev_override()).unwrap();
        expect_eq!(resolved, "dummy::good");
    }

    #[gtest]
    fn unknown_bare_model_is_rejected() {
        let error = resolve_systemone_model("gpt-4o", None, &ModelAliasTable::default())
            .expect_err("chat models are not System One");
        expect_that!(error.to_string(), contains_substring("not supported"));
    }

    #[gtest]
    fn dummy_answers_cover_noul_choice_and_score() {
        let mut questions = Map::new();
        questions.insert(
            "refund".to_string(),
            json!({"type": "noul", "instructions": "refund?"}),
        );
        questions.insert(
            "team".to_string(),
            json!({
                "type": "choice",
                "instructions": "which team",
                "criteria": { "billing": "charges", "technical": "bugs" }
            }),
        );
        questions.insert(
            "urgency".to_string(),
            json!({"type": "score", "instructions": "how urgent", "criteria": ["low", "high"]}),
        );
        let (status, body) = dummy_systemone("good", &questions).unwrap();
        expect_eq!(status, StatusCode::OK);
        expect_eq!(body["answers"]["refund"]["noul"], 0.5);
        expect_eq!(body["answers"]["team"]["choice"], "billing");
        expect_eq!(body["answers"]["urgency"]["score"], 0.0);
        let usage = usage_from_systemone(&body);
        expect_eq!(usage.input_tokens, Some(12));
        expect_eq!(usage.output_tokens, Some(3));
        expect_eq!(
            usage.cost,
            Some(Decimal::from(12) * JEV_INPUT_USD_PER_TOKEN)
        );
    }

    #[gtest]
    fn provider_cost_wins_over_list_price() {
        let body = json!({
            "usage": { "input_tokens": 1_000_000, "output_tokens": 20, "cost": 0.03 }
        });
        let usage = usage_from_systemone(&body);
        expect_eq!(usage.cost, Some(Decimal::from_str_exact("0.03").unwrap()));
        expect_eq!(usage.currency, Some(tensorzero_types::Currency::USD));
    }

    #[gtest]
    fn upstream_url_is_openrouter_systemone() {
        let base = Url::parse("https://openrouter.ai/api/v1").unwrap();
        let url = join_path(&base, "systemone").unwrap();
        expect_eq!(url.as_str(), "https://openrouter.ai/api/v1/systemone");
    }
}

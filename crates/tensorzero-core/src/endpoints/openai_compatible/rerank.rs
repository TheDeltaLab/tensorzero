// Modified by Delta-AI under Apache 2.0
//! OpenAI-compatible `POST /v1/rerank` for Synapse-compatible clients.
//!
//! Callers send `{ model, query, documents }` with `x-synapse-provider: alibaba`.
//! DashScope's compatible-api path is `/v1/reranks` (note the trailing `s`).

use axum::Json;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use url::Url;

use crate::error::{Error, ErrorDetails};
use crate::http::TensorzeroHttpClient;
use crate::model::{SILICONFLOW_DEFAULT_API_ROOT, openai_compatible_shorthand_api_base};
use crate::model_alias::ModelAliasTable;
use crate::utils::gateway::{AppState, AppStateData};

use super::OpenAIStructuredJson;
use super::infer::error_response;
use super::synapse::{
    SynapseRequestContext, resolve_openai_compatible_model, run_with_request_timeout,
    served_by_from_model_name,
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
    let model = match resolve_rerank_model(
        &params.model,
        synapse.provider.as_deref(),
        &config.models.model_aliases,
    ) {
        Ok(model) => model,
        Err(error) => return Ok(error_response(error, false, &synapse)),
    };
    synapse.served_by = Some(served_by_from_model_name(&model));

    let (query, documents, top_n) = match extract_rerank_args(&params) {
        Ok(args) => args,
        Err(error) => return Ok(error_response(error, false, &synapse)),
    };

    let response = match Box::pin(run_with_request_timeout(
        synapse.request_timeout,
        dispatch_rerank(&http_client, &model, &query, &documents, top_n, &params),
    ))
    .await
    {
        Ok(mut response) => {
            synapse.apply_to_response(&mut response);
            response
        }
        Err(error) => error_response(error, false, &synapse),
    };
    Ok(response)
}

/// Provider header wins; otherwise a `[model_aliases]` entry with `task = "rerank"`
/// supplies the head `provider::model` (Synapse bare-name semantics).
fn resolve_rerank_model(
    model: &str,
    provider: Option<&str>,
    aliases: &ModelAliasTable,
) -> Result<String, Error> {
    let resolved = resolve_openai_compatible_model(model, provider)?;
    if resolved.contains("::") {
        return Ok(resolved);
    }
    if let Some(alias) = aliases.resolve(model.trim(), Some("rerank"))
        && let Some(target) = alias.targets.first()
    {
        return Ok(format!("{}::{}", target.provider_type, target.model_name));
    }
    Ok(resolved)
}

fn extract_rerank_args(
    params: &OpenAICompatibleRerankParams,
) -> Result<(String, Vec<String>, Option<u32>), Error> {
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
    Ok((query, documents, top_n))
}

async fn dispatch_rerank(
    http_client: &TensorzeroHttpClient,
    model: &str,
    query: &str,
    documents: &[String],
    top_n: Option<u32>,
    params: &OpenAICompatibleRerankParams,
) -> Result<Response, Error> {
    let (provider, upstream_model) = split_provider_model(model)?;
    if provider == "dummy" {
        return Ok(dummy_rerank(upstream_model, documents, top_n));
    }

    let (url, api_key_env) = rerank_upstream(provider)?;
    let api_key = std::env::var(api_key_env).map_err(|_| {
        Error::new(ErrorDetails::ApiKeyMissing {
            provider_name: provider.to_string(),
            message: format!("{api_key_env} is not set"),
        })
    })?;

    let body = build_upstream_body(upstream_model, query, documents, top_n, &params.extra);

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

    let mut http = (
        StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
        Json(json),
    )
        .into_response();
    if let Ok(value) = HeaderValue::from_str("application/json") {
        http.headers_mut()
            .insert(axum::http::header::CONTENT_TYPE, value);
    }
    Ok(http)
}

fn build_upstream_body(
    upstream_model: &str,
    query: &str,
    documents: &[String],
    top_n: Option<u32>,
    extra: &serde_json::Map<String, Value>,
) -> Value {
    let mut body = serde_json::Map::new();
    body.insert("model".to_string(), json!(upstream_model));
    body.insert("query".to_string(), json!(query));
    body.insert("documents".to_string(), json!(documents));
    if let Some(top_n) = top_n {
        body.insert("top_n".to_string(), json!(top_n));
    }
    for (key, value) in extra {
        if matches!(
            key.as_str(),
            "model" | "query" | "documents" | "top_n" | "input" | "parameters"
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

fn dummy_rerank(model: &str, documents: &[String], top_n: Option<u32>) -> Response {
    if model.starts_with("error") {
        return Error::new(ErrorDetails::InferenceClient {
            message: format!("Error sending request to Dummy provider for model '{model}'."),
            status_code: Some(StatusCode::INTERNAL_SERVER_ERROR),
            provider_type: "dummy".to_string(),
            api_type: crate::inference::types::ApiType::ChatCompletions,
            raw_request: None,
            raw_response: None,
        })
        .into_response_with_raw_entries(true, false);
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
    Json(json!({
        "results": results,
        "usage": { "total_tokens": 0 }
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_cohere_style() {
        let params: OpenAICompatibleRerankParams = serde_json::from_value(json!({
            "model": "qwen3-rerank",
            "query": "capital",
            "documents": ["Paris", "London"],
            "top_n": 1
        }))
        .unwrap();
        let (query, documents, top_n) = extract_rerank_args(&params).unwrap();
        assert_eq!(query, "capital");
        assert_eq!(documents, vec!["Paris", "London"]);
        assert_eq!(top_n, Some(1));
    }

    #[test]
    fn extract_dashscope_style() {
        let params: OpenAICompatibleRerankParams = serde_json::from_value(json!({
            "model": "qwen3-rerank",
            "input": { "query": "capital", "documents": ["Paris"] },
            "parameters": { "top_n": 2 }
        }))
        .unwrap();
        let (query, documents, top_n) = extract_rerank_args(&params).unwrap();
        assert_eq!(query, "capital");
        assert_eq!(documents, vec!["Paris"]);
        assert_eq!(top_n, Some(2));
    }

    #[test]
    fn unwrap_nested_results() {
        let mut json = json!({ "output": { "results": [{ "index": 0, "relevance_score": 1.0 }] } });
        unwrap_dashscope_results(&mut json);
        assert!(json.get("results").is_some());
    }

    #[test]
    fn dummy_scores_preserve_index() {
        let response = dummy_rerank("good", &["a".into(), "b".into()], Some(1));
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[test]
    fn build_upstream_body_rewrites_model_and_keeps_extra() {
        let mut extra = serde_json::Map::new();
        extra.insert("return_documents".to_string(), json!(true));
        extra.insert("query".to_string(), json!("should-not-win"));
        let body = build_upstream_body(
            "qwen3-rerank",
            "capital",
            &["Paris".into()],
            Some(1),
            &extra,
        );
        assert_eq!(body["model"], "qwen3-rerank");
        assert_eq!(body["query"], "capital");
        assert_eq!(body["top_n"], 1);
        assert_eq!(body["return_documents"], true);
    }

    #[test]
    fn resolve_bare_name_via_rerank_alias() {
        use crate::model_alias::{ModelAlias, ModelAliasTarget};
        use std::sync::Arc;
        let aliases = ModelAliasTable {
            aliases: vec![ModelAlias {
                name: Arc::from("qwen3-rerank"),
                task: Some(Arc::from("rerank")),
                targets: vec![ModelAliasTarget {
                    provider_type: Arc::from("alibaba"),
                    model_name: Arc::from("qwen3-rerank"),
                }],
                min_tokens_per_sec: None,
            }],
        };
        assert_eq!(
            resolve_rerank_model("qwen3-rerank", None, &aliases).unwrap(),
            "alibaba::qwen3-rerank"
        );
        assert_eq!(
            resolve_rerank_model("qwen3-rerank", Some("dummy"), &aliases).unwrap(),
            "dummy::qwen3-rerank"
        );
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
}

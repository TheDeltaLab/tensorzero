// Modified by Delta-AI under Apache 2.0
//! Synapse-compatible internal observability helpers (usage CSV, balances).

use axum::Json;
use axum::extract::{Query, State};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::QueryBuilder;
use tracing::instrument;

use crate::error::{Error, ErrorDetails};
use crate::inference::types::ApiType;
use crate::observability_tags::parse_csv_tags;
use crate::utils::gateway::{AppState, AppStateData};
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
pub struct SynapseTimeRangeQuery {
    pub from: DateTime<Utc>,
    pub to: DateTime<Utc>,
    /// Comma-separated `key=value` pairs, same as `x-tensorzero-tags`.
    #[serde(default)]
    pub tags: Option<String>,
    /// Group analytics by this user-tag key (e.g. `env`).
    #[serde(default)]
    pub group_by_tag: Option<String>,
}

#[derive(Debug, sqlx::FromRow)]
struct UsageRow {
    day: DateTime<Utc>,
    model_name: String,
    model_provider_name: String,
    requests: i64,
    input_tokens: i64,
    output_tokens: i64,
    cache_hit_tokens: i64,
}

#[derive(Debug, Serialize, sqlx::FromRow)]
pub struct SynapseAnalyticsRow {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
    pub model_name: String,
    pub requests: i64,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub avg_latency_ms: Option<f64>,
    pub avg_ttft_ms: Option<f64>,
    pub output_tps_excluding_ttft: Option<f64>,
    pub kind: String,
}

#[derive(Debug, Serialize)]
pub struct SynapseAnalyticsResponse {
    pub data: Vec<SynapseAnalyticsRow>,
}

#[derive(Debug, Serialize)]
pub struct SynapseBalancesResponse {
    pub deepseek: Option<Value>,
    pub openrouter: Option<Value>,
}

/// DeepSeek official CNY per million tokens (pro / flash).
fn deepseek_cny_per_million(model: &str) -> Option<(f64, f64, f64)> {
    let name = model.to_ascii_lowercase();
    if name.contains("deepseek-v4-pro") {
        Some((3.0, 0.025, 6.0))
    } else if name.contains("deepseek-v4-flash") {
        Some((1.0, 0.02, 2.0))
    } else {
        None
    }
}

fn million_cost(tokens: i64, rate: f64) -> f64 {
    tokens as f64 / 1_000_000.0 * rate
}

fn require_pool(app_state: &AppStateData) -> Result<&sqlx::PgPool, Error> {
    app_state
        .postgres_connection_info
        .get_pool()
        .ok_or_else(|| {
            Error::new(ErrorDetails::PostgresConnection {
                message: "Postgres is disabled; Synapse usage export requires observability.backend = postgres".to_string(),
            })
        })
}

fn parse_query_tags(raw: Option<&str>) -> Result<HashMap<String, String>, Error> {
    let Some(raw) = raw.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(HashMap::new());
    };
    parse_csv_tags(raw).map_err(|message| {
        Error::new(ErrorDetails::InvalidRequest {
            message: format!("Invalid `tags` query parameter: {message}"),
        })
    })
}

fn push_tag_contains_filters(
    query_builder: &mut QueryBuilder<sqlx::Postgres>,
    tags: &HashMap<String, String>,
) {
    for (key, value) in tags {
        query_builder.push(" AND i.tags @> jsonb_build_object(");
        query_builder.push_bind(key);
        query_builder.push(", ");
        query_builder.push_bind(value);
        query_builder.push(")");
    }
}

/// Handler for `GET /internal/synapse/usage_export`
#[instrument(name = "synapse.usage_export", skip_all)]
pub async fn usage_export_handler(
    State(app_state): AppState,
    Query(params): Query<SynapseTimeRangeQuery>,
) -> Result<Response, Error> {
    if params.to <= params.from {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: "`to` must be after `from`".to_string(),
        }));
    }
    let pool = require_pool(&app_state)?;
    let mut query_builder: QueryBuilder<sqlx::Postgres> = QueryBuilder::new(
        "SELECT date_trunc('day', created_at) as day, \
                model_name, \
                model_provider_name, \
                COUNT(*)::BIGINT as requests, \
                COALESCE(SUM(input_tokens), 0)::BIGINT as input_tokens, \
                COALESCE(SUM(output_tokens), 0)::BIGINT as output_tokens, \
                COALESCE(SUM(provider_cache_read_input_tokens), 0)::BIGINT as cache_hit_tokens \
         FROM tensorzero.model_inferences \
         WHERE created_at >= ",
    );
    query_builder.push_bind(params.from);
    query_builder.push(" AND created_at < ");
    query_builder.push_bind(params.to);
    query_builder.push(" GROUP BY day, model_name, model_provider_name ORDER BY day, model_name");
    let rows: Vec<UsageRow> = query_builder
        .build_query_as()
        .fetch_all(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to export Synapse usage: {e}"),
            })
        })?;

    let mut csv = String::from(
        "date,model,provider,requests,input_tokens,cache_hit_tokens,output_tokens,input_cost_cny,cache_hit_cost_cny,output_cost_cny,total_cost_cny\n",
    );
    for row in rows {
        let (input_rate, cache_rate, output_rate) =
            deepseek_cny_per_million(&row.model_name).unwrap_or((0.0, 0.0, 0.0));
        let input_cost = million_cost(row.input_tokens, input_rate);
        let cache_cost = million_cost(row.cache_hit_tokens, cache_rate);
        let output_cost = million_cost(row.output_tokens, output_rate);
        let total = input_cost + cache_cost + output_cost;
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6}\n",
            row.day.date_naive(),
            row.model_name,
            row.model_provider_name,
            row.requests,
            row.input_tokens,
            row.cache_hit_tokens,
            row.output_tokens,
            input_cost,
            cache_cost,
            output_cost,
            total,
        ));
    }
    Ok(([(header::CONTENT_TYPE, "text/csv; charset=utf-8")], csv).into_response())
}

/// Handler for `GET /internal/synapse/analytics`
#[instrument(name = "synapse.analytics", skip_all)]
pub async fn analytics_handler(
    State(app_state): AppState,
    Query(params): Query<SynapseTimeRangeQuery>,
) -> Result<Json<SynapseAnalyticsResponse>, Error> {
    if params.to <= params.from {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: "`to` must be after `from`".to_string(),
        }));
    }
    let pool = require_pool(&app_state)?;
    let tag_filter = parse_query_tags(params.tags.as_deref())?;
    let group_by_tag = params
        .group_by_tag
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let join_inferences = !tag_filter.is_empty() || group_by_tag.is_some();

    let mut query_builder: QueryBuilder<sqlx::Postgres> = QueryBuilder::new("SELECT ");
    if let Some(key) = &group_by_tag {
        query_builder.push("i.tags ->> ");
        query_builder.push_bind(key);
        query_builder.push(" as tag, ");
    } else {
        query_builder.push("NULL::TEXT as tag, ");
    }
    query_builder.push(
        "mi.model_name as model_name, \
                COUNT(*)::BIGINT as requests, \
                COALESCE(SUM(mi.input_tokens), 0)::BIGINT as input_tokens, \
                COALESCE(SUM(mi.output_tokens), 0)::BIGINT as output_tokens, \
                AVG(mi.response_time_ms)::FLOAT8 as avg_latency_ms, \
                AVG(mi.ttft_ms)::FLOAT8 as avg_ttft_ms, \
                CASE WHEN SUM(GREATEST(mi.response_time_ms - COALESCE(mi.ttft_ms, 0), 1)) > 0 \
                     THEN (COALESCE(SUM(mi.output_tokens), 0)::FLOAT8 * 1000.0) \
                          / SUM(GREATEST(mi.response_time_ms - COALESCE(mi.ttft_ms, 0), 1))::FLOAT8 \
                     ELSE 0 END as output_tps_excluding_ttft, \
                CASE WHEN COALESCE(SUM(mi.output_tokens), 0) = 0 THEN 'embedding' ELSE 'chat' END as kind \
         FROM tensorzero.model_inferences mi",
    );
    if join_inferences {
        query_builder.push(
            " INNER JOIN ( \
                SELECT id, tags FROM tensorzero.chat_inferences \
                UNION ALL \
                SELECT id, tags FROM tensorzero.json_inferences \
              ) i ON mi.inference_id = i.id",
        );
    }
    query_builder.push(" WHERE mi.created_at >= ");
    query_builder.push_bind(params.from);
    query_builder.push(" AND mi.created_at < ");
    query_builder.push_bind(params.to);
    if join_inferences {
        push_tag_contains_filters(&mut query_builder, &tag_filter);
    }
    if group_by_tag.is_some() {
        query_builder.push(" GROUP BY 1, mi.model_name ORDER BY mi.model_name, 1");
    } else {
        query_builder.push(" GROUP BY mi.model_name ORDER BY mi.model_name");
    }
    let data: Vec<SynapseAnalyticsRow> = query_builder
        .build_query_as()
        .fetch_all(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to load Synapse analytics: {e}"),
            })
        })?;
    Ok(Json(SynapseAnalyticsResponse { data }))
}

/// Handler for `GET /internal/synapse/balances`
#[instrument(name = "synapse.balances", skip_all)]
pub async fn balances_handler(
    State(app_state): AppState,
) -> Result<Json<SynapseBalancesResponse>, Error> {
    let http = &app_state.http_client;
    let deepseek = match std::env::var("DEEPSEEK_API_KEY") {
        Ok(key) if !key.is_empty() => Some(
            http.get("https://api.deepseek.com/user/balance")
                .bearer_auth(&key)
                .send()
                .await
                .map_err(|e| {
                    Error::new(ErrorDetails::InferenceClient {
                        status_code: None,
                        message: format!("DeepSeek balance request failed: {e}"),
                        provider_type: "deepseek".to_string(),
                        api_type: ApiType::Other,
                        raw_request: None,
                        raw_response: None,
                    })
                })?
                .json::<Value>()
                .await
                .map_err(|e| {
                    Error::new(ErrorDetails::InferenceClient {
                        status_code: None,
                        message: format!("DeepSeek balance JSON failed: {e}"),
                        provider_type: "deepseek".to_string(),
                        api_type: ApiType::Other,
                        raw_request: None,
                        raw_response: None,
                    })
                })?,
        ),
        _ => None,
    };
    let openrouter = match std::env::var("OPENROUTER_API_KEY") {
        Ok(key) if !key.is_empty() => Some(
            http.get("https://openrouter.ai/api/v1/credits")
                .bearer_auth(&key)
                .send()
                .await
                .map_err(|e| {
                    Error::new(ErrorDetails::InferenceClient {
                        status_code: None,
                        message: format!("OpenRouter credits request failed: {e}"),
                        provider_type: "openrouter".to_string(),
                        api_type: ApiType::Other,
                        raw_request: None,
                        raw_response: None,
                    })
                })?
                .json::<Value>()
                .await
                .map_err(|e| {
                    Error::new(ErrorDetails::InferenceClient {
                        status_code: None,
                        message: format!("OpenRouter credits JSON failed: {e}"),
                        provider_type: "openrouter".to_string(),
                        api_type: ApiType::Other,
                        raw_request: None,
                        raw_response: None,
                    })
                })?,
        ),
        _ => None,
    };
    Ok(Json(SynapseBalancesResponse {
        deepseek,
        openrouter,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prices_pro_and_flash() {
        assert_eq!(
            deepseek_cny_per_million("deepseek-v4-pro"),
            Some((3.0, 0.025, 6.0))
        );
        assert_eq!(
            deepseek_cny_per_million("deepseek::deepseek-v4-flash"),
            Some((1.0, 0.02, 2.0))
        );
        assert_eq!(deepseek_cny_per_million("glm-5"), None);
    }
}

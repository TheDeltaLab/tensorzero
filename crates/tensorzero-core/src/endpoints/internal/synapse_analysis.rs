// Modified by Delta-AI under Apache 2.0
//! Synapse-compatible Analysis dashboard aggregations over Postgres observability.

use axum::Json;
use axum::extract::{Query, State};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use sqlx::QueryBuilder;
use tracing::instrument;

use crate::error::{Error, ErrorDetails};
use crate::function::{EMBEDDING_FUNCTION_NAME, RERANK_FUNCTION_NAME};
use crate::observability_tags::API_KEY_PUBLIC_ID_TAG;
use crate::utils::gateway::{AppState, AppStateData};

const SUCCESS_STATUS_SQL: &str =
    "COALESCE(i.tags->>'tensorzero::status_code', '200') ~ '^2[0-9]{2}$'";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisRange {
    FifteenMinutes,
    OneHour,
    TwentyFourHours,
    SevenDays,
    ThirtyDays,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisKind {
    Chat,
    Embedding,
}

#[derive(Debug, Deserialize)]
pub struct AnalysisQuery {
    #[serde(default = "default_range")]
    pub range: String,
    #[serde(default = "default_kind")]
    pub kind: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub cache_miss_only: bool,
}

fn default_range() -> String {
    "24h".to_string()
}

fn default_kind() -> String {
    "chat".to_string()
}

#[derive(Debug, Serialize)]
pub struct AnalysisProviderStats {
    pub provider: String,
    pub count: i64,
    pub percentage: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnalysisModelStats {
    pub model: String,
    pub provider: String,
    pub count: i64,
    pub avg_latency: Option<f64>,
    pub p50: Option<f64>,
    pub p90: Option<f64>,
    pub p99: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct AnalysisCountPoint {
    pub date: String,
    pub count: i64,
}

#[derive(Debug, Serialize)]
pub struct AnalysisPercentilePoint {
    pub date: String,
    pub p50: Option<f64>,
    pub p90: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p99: Option<f64>,
    pub avg: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct AnalysisTokenPoint {
    pub date: String,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub total_tokens: i64,
    pub count: i64,
}

#[derive(Debug, Serialize)]
pub struct AnalysisResponse {
    pub total_requests: i64,
    pub total_responses: i64,
    pub success_rate: f64,
    pub cache_hit_rate: f64,
    pub input_cache_hit_rate: f64,
    pub cache_read_input_tokens: i64,
    pub unique_providers: i64,
    pub unique_models: i64,
    pub avg_latency: Option<f64>,
    pub total_tokens: i64,
    pub total_input_tokens: i64,
    pub total_output_tokens: i64,
    pub provider_stats: Vec<AnalysisProviderStats>,
    pub model_stats: Vec<AnalysisModelStats>,
    pub model_latency_stats: Vec<AnalysisModelStats>,
    pub token_usage_over_time: Vec<AnalysisTokenPoint>,
    pub requests_over_time: Vec<AnalysisCountPoint>,
    pub latency_over_time: Vec<AnalysisPercentilePoint>,
    pub ttft_over_time: Vec<AnalysisPercentilePoint>,
    pub output_tps_over_time: Vec<AnalysisPercentilePoint>,
}

#[derive(sqlx::FromRow)]
struct TotalsRow {
    total_requests: i64,
    total_responses: i64,
    cached_requests: i64,
    unique_providers: i64,
    unique_models: i64,
    avg_latency: Option<f64>,
    total_input_tokens: i64,
    total_output_tokens: i64,
    cache_read_input_tokens: i64,
}

#[derive(sqlx::FromRow)]
struct ProviderRow {
    provider: String,
    count: i64,
}

#[derive(sqlx::FromRow)]
struct ModelRow {
    model: String,
    provider: String,
    count: i64,
    avg: Option<f64>,
    p50: Option<f64>,
    p90: Option<f64>,
    p99: Option<f64>,
}

#[derive(sqlx::FromRow)]
struct SeriesRow {
    bucket: DateTime<Utc>,
    requests: i64,
    input_tokens: i64,
    output_tokens: i64,
    avg_latency: Option<f64>,
    latency_p50: Option<f64>,
    latency_p90: Option<f64>,
    latency_p99: Option<f64>,
    ttft_avg: Option<f64>,
    ttft_p50: Option<f64>,
    ttft_p90: Option<f64>,
    tps_avg: Option<f64>,
    tps_p50: Option<f64>,
    tps_p90: Option<f64>,
}

struct AnalysisParams {
    range: AnalysisRange,
    kind: AnalysisKind,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
    api_key: Option<String>,
    model: Option<String>,
    cache_miss_only: bool,
}

impl AnalysisRange {
    pub fn parse(raw: &str) -> Result<Self, Error> {
        match raw {
            "15m" => Ok(Self::FifteenMinutes),
            "1h" => Ok(Self::OneHour),
            "24h" => Ok(Self::TwentyFourHours),
            "7d" => Ok(Self::SevenDays),
            "30d" => Ok(Self::ThirtyDays),
            other => Err(Error::new(ErrorDetails::InvalidRequest {
                message: format!(
                    "Invalid `range` `{other}`. Must be one of: 15m, 1h, 24h, 7d, 30d"
                ),
            })),
        }
    }

    fn duration(self) -> Duration {
        match self {
            Self::FifteenMinutes => Duration::minutes(15),
            Self::OneHour => Duration::hours(1),
            Self::TwentyFourHours => Duration::hours(24),
            Self::SevenDays => Duration::days(7),
            Self::ThirtyDays => Duration::days(30),
        }
    }

    fn trunc_unit(self) -> &'static str {
        match self {
            Self::FifteenMinutes | Self::OneHour => "minute",
            Self::TwentyFourHours => "hour",
            Self::SevenDays | Self::ThirtyDays => "day",
        }
    }

    pub fn format_bucket(self, ts: DateTime<Utc>) -> String {
        match self {
            Self::FifteenMinutes | Self::OneHour => ts.format("%Y-%m-%dT%H:%M:00Z").to_string(),
            Self::TwentyFourHours => ts.format("%Y-%m-%dT%H:00:00Z").to_string(),
            Self::SevenDays | Self::ThirtyDays => ts.format("%Y-%m-%d").to_string(),
        }
    }
}

impl AnalysisKind {
    fn parse(raw: &str) -> Result<Self, Error> {
        match raw {
            "chat" => Ok(Self::Chat),
            "embedding" => Ok(Self::Embedding),
            other => Err(Error::new(ErrorDetails::InvalidRequest {
                message: format!("Invalid `kind` `{other}`. Must be `chat` or `embedding`"),
            })),
        }
    }
}

/// Input cache hit rate as a percentage: cache-read input tokens / total input tokens.
fn input_cache_hit_rate_pct(cache_read_input_tokens: i64, total_input_tokens: i64) -> f64 {
    if total_input_tokens > 0 {
        (cache_read_input_tokens as f64 / total_input_tokens as f64) * 100.0
    } else {
        0.0
    }
}

fn require_pool(app_state: &AppStateData) -> Result<&sqlx::PgPool, Error> {
    app_state
        .postgres_connection_info
        .get_pool()
        .ok_or_else(|| {
            Error::new(ErrorDetails::PostgresConnection {
                message: "Postgres is disabled; Analysis requires observability.backend = postgres"
                    .to_string(),
            })
        })
}

fn push_from_and_filters(
    query_builder: &mut QueryBuilder<sqlx::Postgres>,
    params: &AnalysisParams,
    successful_only: bool,
) {
    query_builder.push(
        " FROM tensorzero.model_inferences mi \
         INNER JOIN ( \
            SELECT id, tags, function_name FROM tensorzero.chat_inferences \
            UNION ALL \
            SELECT id, tags, function_name FROM tensorzero.json_inferences \
         ) i ON mi.inference_id = i.id \
         WHERE mi.created_at >= ",
    );
    query_builder.push_bind(params.from);
    query_builder.push(" AND mi.created_at < ");
    query_builder.push_bind(params.to);
    match params.kind {
        AnalysisKind::Chat => {
            query_builder.push(" AND i.function_name <> ");
            query_builder.push_bind(EMBEDDING_FUNCTION_NAME);
            query_builder.push(" AND i.function_name <> ");
            query_builder.push_bind(RERANK_FUNCTION_NAME);
        }
        AnalysisKind::Embedding => {
            query_builder.push(" AND i.function_name = ");
            query_builder.push_bind(EMBEDDING_FUNCTION_NAME);
        }
    }
    if let Some(api_key) = params
        .api_key
        .as_deref()
        .map(str::trim)
        .filter(|v| !v.is_empty())
    {
        query_builder.push(" AND i.tags ->> ");
        query_builder.push_bind(API_KEY_PUBLIC_ID_TAG);
        query_builder.push(" = ");
        query_builder.push_bind(api_key);
    }
    if let Some(model) = params
        .model
        .as_deref()
        .map(str::trim)
        .filter(|v| !v.is_empty())
    {
        query_builder.push(" AND mi.model_name = ");
        query_builder.push_bind(model);
    }
    if params.cache_miss_only {
        query_builder.push(" AND mi.cached = FALSE");
    }
    if successful_only {
        query_builder.push(" AND ");
        query_builder.push(SUCCESS_STATUS_SQL);
    }
}

async fn fetch_totals(pool: &sqlx::PgPool, params: &AnalysisParams) -> Result<TotalsRow, Error> {
    let mut query_builder: QueryBuilder<sqlx::Postgres> = QueryBuilder::new(format!(
        "SELECT \
            COUNT(*) FILTER (WHERE {SUCCESS_STATUS_SQL})::BIGINT as total_requests, \
            COUNT(*)::BIGINT as total_responses, \
            COUNT(*) FILTER (WHERE mi.cached AND {SUCCESS_STATUS_SQL})::BIGINT as cached_requests, \
            COUNT(DISTINCT mi.model_provider_name) FILTER (WHERE {SUCCESS_STATUS_SQL})::BIGINT as unique_providers, \
            COUNT(DISTINCT mi.model_name) FILTER (WHERE {SUCCESS_STATUS_SQL})::BIGINT as unique_models, \
            AVG(mi.response_time_ms) FILTER (WHERE {SUCCESS_STATUS_SQL})::FLOAT8 as avg_latency, \
            COALESCE(SUM(mi.input_tokens) FILTER (WHERE {SUCCESS_STATUS_SQL}), 0)::BIGINT as total_input_tokens, \
            COALESCE(SUM(mi.output_tokens) FILTER (WHERE {SUCCESS_STATUS_SQL}), 0)::BIGINT as total_output_tokens, \
            COALESCE(SUM(mi.provider_cache_read_input_tokens) FILTER (WHERE {SUCCESS_STATUS_SQL}), 0)::BIGINT as cache_read_input_tokens"
    ));
    push_from_and_filters(&mut query_builder, params, false);
    query_builder
        .build_query_as()
        .fetch_one(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to load Analysis totals: {e}"),
            })
        })
}

async fn fetch_providers(
    pool: &sqlx::PgPool,
    params: &AnalysisParams,
) -> Result<Vec<ProviderRow>, Error> {
    let mut query_builder: QueryBuilder<sqlx::Postgres> =
        QueryBuilder::new("SELECT mi.model_provider_name as provider, COUNT(*)::BIGINT as count");
    push_from_and_filters(&mut query_builder, params, true);
    query_builder.push(" GROUP BY mi.model_provider_name ORDER BY count DESC, provider");
    query_builder
        .build_query_as()
        .fetch_all(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to load Analysis provider stats: {e}"),
            })
        })
}

async fn fetch_models(
    pool: &sqlx::PgPool,
    params: &AnalysisParams,
) -> Result<Vec<ModelRow>, Error> {
    let mut query_builder: QueryBuilder<sqlx::Postgres> = QueryBuilder::new(
        "SELECT mi.model_name as model, \
                mi.model_provider_name as provider, \
                COUNT(*)::BIGINT as count, \
                AVG(mi.response_time_ms)::FLOAT8 as avg, \
                percentile_cont(0.5) WITHIN GROUP (ORDER BY mi.response_time_ms) \
                    FILTER (WHERE mi.response_time_ms IS NOT NULL) as p50, \
                percentile_cont(0.9) WITHIN GROUP (ORDER BY mi.response_time_ms) \
                    FILTER (WHERE mi.response_time_ms IS NOT NULL) as p90, \
                percentile_cont(0.99) WITHIN GROUP (ORDER BY mi.response_time_ms) \
                    FILTER (WHERE mi.response_time_ms IS NOT NULL) as p99",
    );
    push_from_and_filters(&mut query_builder, params, true);
    query_builder.push(
        " GROUP BY mi.model_name, mi.model_provider_name ORDER BY count DESC, model, provider",
    );
    query_builder
        .build_query_as()
        .fetch_all(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to load Analysis model stats: {e}"),
            })
        })
}

async fn fetch_series(
    pool: &sqlx::PgPool,
    params: &AnalysisParams,
) -> Result<Vec<SeriesRow>, Error> {
    let trunc = params.range.trunc_unit();
    let tps_expr = "(mi.output_tokens::FLOAT8 * 1000.0) \
         / NULLIF((mi.response_time_ms - COALESCE(mi.ttft_ms, 0)), 0)::FLOAT8";
    let mut query_builder: QueryBuilder<sqlx::Postgres> = QueryBuilder::new(format!(
        "SELECT date_trunc('{trunc}', mi.created_at) as bucket, \
                COUNT(*)::BIGINT as requests, \
                COALESCE(SUM(mi.input_tokens), 0)::BIGINT as input_tokens, \
                COALESCE(SUM(mi.output_tokens), 0)::BIGINT as output_tokens, \
                AVG(mi.response_time_ms)::FLOAT8 as avg_latency, \
                percentile_cont(0.5) WITHIN GROUP (ORDER BY mi.response_time_ms) \
                    FILTER (WHERE mi.response_time_ms IS NOT NULL) as latency_p50, \
                percentile_cont(0.9) WITHIN GROUP (ORDER BY mi.response_time_ms) \
                    FILTER (WHERE mi.response_time_ms IS NOT NULL) as latency_p90, \
                percentile_cont(0.99) WITHIN GROUP (ORDER BY mi.response_time_ms) \
                    FILTER (WHERE mi.response_time_ms IS NOT NULL) as latency_p99, \
                AVG(mi.ttft_ms)::FLOAT8 as ttft_avg, \
                percentile_cont(0.5) WITHIN GROUP (ORDER BY mi.ttft_ms) \
                    FILTER (WHERE mi.ttft_ms IS NOT NULL) as ttft_p50, \
                percentile_cont(0.9) WITHIN GROUP (ORDER BY mi.ttft_ms) \
                    FILTER (WHERE mi.ttft_ms IS NOT NULL) as ttft_p90, \
                AVG({tps_expr}) as tps_avg, \
                percentile_cont(0.5) WITHIN GROUP (ORDER BY {tps_expr}) \
                    FILTER (WHERE mi.output_tokens > 0 AND mi.response_time_ms IS NOT NULL) as tps_p50, \
                percentile_cont(0.9) WITHIN GROUP (ORDER BY {tps_expr}) \
                    FILTER (WHERE mi.output_tokens > 0 AND mi.response_time_ms IS NOT NULL) as tps_p90"
    ));
    push_from_and_filters(&mut query_builder, params, true);
    query_builder.push(format!(
        " GROUP BY date_trunc('{trunc}', mi.created_at) ORDER BY 1"
    ));
    query_builder
        .build_query_as()
        .fetch_all(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to load Analysis timeseries: {e}"),
            })
        })
}

/// Handler for `GET /internal/synapse/analysis`
#[instrument(name = "synapse.analysis", skip_all)]
pub async fn analysis_handler(
    State(app_state): AppState,
    Query(query): Query<AnalysisQuery>,
) -> Result<Json<AnalysisResponse>, Error> {
    let range = AnalysisRange::parse(&query.range)?;
    let kind = AnalysisKind::parse(&query.kind)?;
    let to = Utc::now();
    let from = to - range.duration();
    let params = AnalysisParams {
        range,
        kind,
        from,
        to,
        api_key: query.api_key,
        model: query.model,
        cache_miss_only: query.cache_miss_only,
    };
    let pool = require_pool(&app_state)?;

    let (totals, providers, models, series) = tokio::try_join!(
        fetch_totals(pool, &params),
        fetch_providers(pool, &params),
        fetch_models(pool, &params),
        fetch_series(pool, &params),
    )?;

    let total_requests = totals.total_requests;
    let success_rate = if totals.total_responses > 0 {
        (total_requests as f64 / totals.total_responses as f64) * 100.0
    } else {
        0.0
    };
    let cache_hit_rate = if total_requests > 0 {
        (totals.cached_requests as f64 / total_requests as f64) * 100.0
    } else {
        0.0
    };
    let input_cache_hit_rate =
        input_cache_hit_rate_pct(totals.cache_read_input_tokens, totals.total_input_tokens);
    let provider_stats = providers
        .into_iter()
        .map(|row| AnalysisProviderStats {
            percentage: if total_requests > 0 {
                (row.count as f64 / total_requests as f64) * 100.0
            } else {
                0.0
            },
            provider: row.provider,
            count: row.count,
        })
        .collect();
    let model_stats: Vec<AnalysisModelStats> = models
        .into_iter()
        .map(|row| AnalysisModelStats {
            model: row.model,
            provider: row.provider,
            count: row.count,
            avg_latency: row.avg,
            p50: row.p50,
            p90: row.p90,
            p99: row.p99,
        })
        .collect();

    let mut requests_over_time = Vec::with_capacity(series.len());
    let mut token_usage_over_time = Vec::with_capacity(series.len());
    let mut latency_over_time = Vec::with_capacity(series.len());
    let mut ttft_over_time = Vec::new();
    let mut output_tps_over_time = Vec::new();
    for row in series {
        let date = range.format_bucket(row.bucket);
        requests_over_time.push(AnalysisCountPoint {
            date: date.clone(),
            count: row.requests,
        });
        token_usage_over_time.push(AnalysisTokenPoint {
            date: date.clone(),
            input_tokens: row.input_tokens,
            output_tokens: row.output_tokens,
            total_tokens: row.input_tokens + row.output_tokens,
            count: row.requests,
        });
        latency_over_time.push(AnalysisPercentilePoint {
            date: date.clone(),
            p50: row.latency_p50,
            p90: row.latency_p90,
            p99: row.latency_p99,
            avg: row.avg_latency,
        });
        if row.ttft_p50.is_some() || row.ttft_p90.is_some() || row.ttft_avg.is_some() {
            ttft_over_time.push(AnalysisPercentilePoint {
                date: date.clone(),
                p50: row.ttft_p50,
                p90: row.ttft_p90,
                p99: None,
                avg: row.ttft_avg,
            });
        }
        if row.tps_p50.is_some() || row.tps_p90.is_some() || row.tps_avg.is_some() {
            output_tps_over_time.push(AnalysisPercentilePoint {
                date,
                p50: row.tps_p50,
                p90: row.tps_p90,
                p99: None,
                avg: row.tps_avg,
            });
        }
    }

    Ok(Json(AnalysisResponse {
        total_requests,
        total_responses: totals.total_responses,
        success_rate,
        cache_hit_rate,
        input_cache_hit_rate,
        cache_read_input_tokens: totals.cache_read_input_tokens,
        unique_providers: totals.unique_providers,
        unique_models: totals.unique_models,
        avg_latency: totals.avg_latency,
        total_tokens: totals.total_input_tokens + totals.total_output_tokens,
        total_input_tokens: totals.total_input_tokens,
        total_output_tokens: totals.total_output_tokens,
        provider_stats,
        model_latency_stats: model_stats.clone(),
        model_stats,
        token_usage_over_time,
        requests_over_time,
        latency_over_time,
        ttft_over_time,
        output_tps_over_time,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use googletest::prelude::*;

    #[gtest]
    fn input_cache_hit_rate_is_cache_read_over_input() {
        expect_eq!(input_cache_hit_rate_pct(0, 0), 0.0);
        expect_eq!(input_cache_hit_rate_pct(25, 100), 25.0);
        expect_eq!(input_cache_hit_rate_pct(100, 100), 100.0);
    }

    #[gtest]
    fn parses_ranges_and_bucket_formats() {
        expect_eq!(
            AnalysisRange::parse("15m").expect("15m"),
            AnalysisRange::FifteenMinutes
        );
        expect_eq!(
            AnalysisRange::parse("30d").expect("30d"),
            AnalysisRange::ThirtyDays
        );
        expect_true!(AnalysisRange::parse("year").is_err());

        let ts = Utc
            .with_ymd_and_hms(2026, 8, 21, 5, 7, 9)
            .single()
            .expect("timestamp");
        expect_eq!(
            AnalysisRange::FifteenMinutes.format_bucket(ts),
            "2026-08-21T05:07:00Z"
        );
        expect_eq!(
            AnalysisRange::TwentyFourHours.format_bucket(ts),
            "2026-08-21T05:00:00Z"
        );
        expect_eq!(AnalysisRange::SevenDays.format_bucket(ts), "2026-08-21");
    }

    #[gtest]
    fn parses_kind() {
        expect_eq!(
            AnalysisKind::parse("chat").expect("chat"),
            AnalysisKind::Chat
        );
        expect_eq!(
            AnalysisKind::parse("embedding").expect("embedding"),
            AnalysisKind::Embedding
        );
        expect_true!(AnalysisKind::parse("rerank").is_err());
    }
}

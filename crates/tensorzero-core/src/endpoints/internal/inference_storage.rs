// Modified by Delta-AI under Apache 2.0
//! Internal endpoints for inference storage stats and retention configuration.

use axum::Json;
use axum::extract::State;
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::db::postgres::inference_storage::InferenceRetentionRow;
use crate::error::{Error, ErrorDetails};
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};

/// Per-table storage statistics for an inference-related Postgres table.
#[derive(ts_rs::TS, Debug, Serialize, Deserialize)]
#[ts(export)]
pub struct InferenceTableStorageStats {
    pub name: String,
    pub total_bytes: i64,
    pub estimated_rows: i64,
    pub partition_count: i64,
}

/// Inference retention configuration.
///
/// `metadata_pinned_by_toml` / `data_pinned_by_toml` indicate that the value
/// is set in `tensorzero.toml` (`[postgres]` section), in which case gateway
/// startup overwrites the database value with the TOML value.
#[derive(ts_rs::TS, Debug, Serialize, Deserialize)]
#[ts(export, optional_fields)]
pub struct InferenceRetentionConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata_retention_days: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_retention_days: Option<u32>,
    pub metadata_pinned_by_toml: bool,
    pub data_pinned_by_toml: bool,
}

/// Response for `GET /internal/inference_storage/stats`.
#[derive(ts_rs::TS, Debug, Serialize, Deserialize)]
#[ts(export)]
pub struct InferenceStorageStatsResponse {
    pub tables: Vec<InferenceTableStorageStats>,
    pub retention: InferenceRetentionConfig,
}

/// Request body for `POST /internal/inference_storage/retention`.
/// A `null` (or absent) value means "keep forever" and deletes the key.
#[derive(ts_rs::TS, Debug, Serialize, Deserialize)]
#[ts(export, optional_fields)]
pub struct UpdateInferenceRetentionRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata_retention_days: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_retention_days: Option<u32>,
}

fn postgres_required(app_state: &AppStateData) -> Result<(), Error> {
    if app_state.postgres_connection_info.get_pool().is_none() {
        return Err(Error::new(ErrorDetails::PostgresConnection {
            message: "Postgres is required for inference storage management".to_string(),
        }));
    }
    Ok(())
}

fn retention_config(
    app_state: &AppStateData,
    retention: InferenceRetentionRow,
) -> InferenceRetentionConfig {
    InferenceRetentionConfig {
        metadata_retention_days: retention.metadata_retention_days,
        data_retention_days: retention.data_retention_days,
        metadata_pinned_by_toml: app_state
            .config
            .postgres
            .inference_metadata_retention_days
            .is_some(),
        data_pinned_by_toml: app_state
            .config
            .postgres
            .inference_data_retention_days
            .is_some(),
    }
}

/// Handler for `GET /internal/inference_storage/stats`
#[instrument(name = "get_inference_storage_stats", skip_all)]
pub async fn get_inference_storage_stats_handler(
    State(app_state): AppState,
) -> Result<Json<InferenceStorageStatsResponse>, Error> {
    postgres_required(&app_state)?;

    let tables = app_state
        .postgres_connection_info
        .get_inference_storage_stats()
        .await?
        .into_iter()
        .map(|row| InferenceTableStorageStats {
            name: row.name,
            total_bytes: row.total_bytes,
            estimated_rows: row.estimated_rows,
            partition_count: row.partition_count,
        })
        .collect();

    let retention = app_state
        .postgres_connection_info
        .get_inference_retention()
        .await?;

    Ok(Json(InferenceStorageStatsResponse {
        tables,
        retention: retention_config(&app_state, retention),
    }))
}

/// Handler for `POST /internal/inference_storage/retention`
#[instrument(name = "update_inference_retention", skip_all)]
pub async fn update_inference_retention_handler(
    State(app_state): AppState,
    StructuredJson(request): StructuredJson<UpdateInferenceRetentionRequest>,
) -> Result<Json<InferenceRetentionConfig>, Error> {
    postgres_required(&app_state)?;

    if let Some(days) = request.metadata_retention_days
        && days < 1
    {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: "`metadata_retention_days` must be at least 1 (or null to keep forever)"
                .to_string(),
        }));
    }
    if let Some(days) = request.data_retention_days
        && days < 1
    {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: "`data_retention_days` must be at least 1 (or null to keep forever)"
                .to_string(),
        }));
    }

    app_state
        .postgres_connection_info
        .set_inference_retention(request.metadata_retention_days, request.data_retention_days)
        .await?;

    let retention = app_state
        .postgres_connection_info
        .get_inference_retention()
        .await?;

    Ok(Json(retention_config(&app_state, retention)))
}

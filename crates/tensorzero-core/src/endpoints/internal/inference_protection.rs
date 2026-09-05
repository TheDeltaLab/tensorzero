// Modified by Delta-AI under Apache 2.0
//! Internal endpoints for per-inference protection from retention cleanup.

use axum::Json;
use axum::extract::{Path, State};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::instrument;
use uuid::Uuid;

use crate::error::{Error, ErrorDetails};
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};

const MAX_PROTECTION_LOOKUP_IDS: usize = 1000;

/// Request body for `POST /internal/inferences/{inference_id}/protection`.
#[derive(ts_rs::TS, Debug, Serialize, Deserialize)]
#[ts(export)]
pub struct SetInferenceProtectionRequest {
    pub protected: bool,
}

/// Response for `POST /internal/inferences/{inference_id}/protection`.
#[derive(ts_rs::TS, Debug, Serialize, Deserialize)]
#[ts(export, optional_fields)]
pub struct InferenceProtectionResponse {
    pub inference_id: Uuid,
    pub function_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protected_at: Option<DateTime<Utc>>,
}

/// Request body for `POST /internal/inferences/protection`.
#[derive(ts_rs::TS, Debug, Serialize, Deserialize)]
#[ts(export)]
pub struct GetInferencesProtectionRequest {
    pub ids: Vec<Uuid>,
}

/// Protection state of a single inference.
#[derive(ts_rs::TS, Debug, Serialize, Deserialize)]
#[ts(export)]
pub struct InferenceProtectionEntry {
    pub id: Uuid,
    pub function_type: String,
    pub protected_at: DateTime<Utc>,
}

/// Response for `POST /internal/inferences/protection`.
#[derive(ts_rs::TS, Debug, Serialize, Deserialize)]
#[ts(export)]
pub struct InferencesProtectionResponse {
    pub protection: Vec<InferenceProtectionEntry>,
}

fn postgres_required(app_state: &AppStateData) -> Result<(), Error> {
    if app_state.postgres_connection_info.get_pool().is_none() {
        return Err(Error::new(ErrorDetails::PostgresConnection {
            message: "Postgres is required for inference protection".to_string(),
        }));
    }
    Ok(())
}

/// Handler for `POST /internal/inferences/{inference_id}/protection`
#[instrument(name = "set_inference_protection", skip_all)]
pub async fn set_inference_protection_handler(
    State(app_state): AppState,
    Path(inference_id): Path<Uuid>,
    StructuredJson(request): StructuredJson<SetInferenceProtectionRequest>,
) -> Result<Json<InferenceProtectionResponse>, Error> {
    postgres_required(&app_state)?;

    let (function_type, protected_at) = app_state
        .postgres_connection_info
        .set_inference_protection(inference_id, request.protected)
        .await?;

    Ok(Json(InferenceProtectionResponse {
        inference_id,
        function_type,
        protected_at,
    }))
}

/// Handler for `POST /internal/inferences/protection`
#[instrument(name = "get_inferences_protection", skip_all)]
pub async fn get_inferences_protection_handler(
    State(app_state): AppState,
    StructuredJson(request): StructuredJson<GetInferencesProtectionRequest>,
) -> Result<Json<InferencesProtectionResponse>, Error> {
    postgres_required(&app_state)?;

    if request.ids.len() > MAX_PROTECTION_LOOKUP_IDS {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!(
                "Too many inference ids: got {}, maximum is {MAX_PROTECTION_LOOKUP_IDS}",
                request.ids.len()
            ),
        }));
    }

    let protection = app_state
        .postgres_connection_info
        .get_inferences_protection(&request.ids)
        .await?
        .into_iter()
        .map(|row| InferenceProtectionEntry {
            id: row.id,
            function_type: row.function_type,
            protected_at: row.protected_at,
        })
        .collect();

    Ok(Json(InferencesProtectionResponse { protection }))
}

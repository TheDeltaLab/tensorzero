// Modified by Delta-AI under Apache 2.0
//! API keys shown in the Inferences filter dropdown.
//!
//! Combines TensorZero native keys, imported Synapse keys, and distinct
//! `tensorzero::api_key_public_id` tags already stored on inferences.

use axum::Json;
use axum::extract::State;
use serde::Serialize;
use tracing::instrument;

use crate::error::{Error, ErrorDetails};
use crate::utils::gateway::AppState;

const API_KEY_SELECT_LIMIT: i64 = 1000;

#[derive(Debug, Serialize, sqlx::FromRow)]
pub struct InferenceApiKeyOption {
    pub public_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub disabled: bool,
}

#[derive(Debug, Serialize)]
pub struct ListInferenceApiKeysResponse {
    pub api_keys: Vec<InferenceApiKeyOption>,
}

/// Handler for `GET /internal/inference_api_keys`
#[instrument(name = "list_inference_api_keys", skip_all)]
pub async fn list_inference_api_keys_handler(
    State(app_state): AppState,
) -> Result<Json<ListInferenceApiKeysResponse>, Error> {
    let Some(pool) = app_state.postgres_connection_info.get_pool() else {
        return Ok(Json(ListInferenceApiKeysResponse { api_keys: vec![] }));
    };

    let api_keys: Vec<InferenceApiKeyOption> = sqlx::query_as(
        r"
        SELECT DISTINCT ON (public_id)
            public_id,
            description,
            disabled
        FROM (
            SELECT btrim(public_id::text) AS public_id,
                   description,
                   (disabled_at IS NOT NULL) AS disabled,
                   0 AS rank
            FROM tensorzero_auth_api_key
            UNION ALL
            SELECT btrim(public_id::text),
                   description,
                   (disabled_at IS NOT NULL),
                   0
            FROM tensorzero_auth_synapse_api_key
            UNION ALL
            SELECT DISTINCT tags->>'tensorzero::api_key_public_id',
                   NULL,
                   FALSE,
                   1
            FROM tensorzero.chat_inferences
            WHERE tags ? 'tensorzero::api_key_public_id'
              AND COALESCE(tags->>'tensorzero::api_key_public_id', '') <> ''
            UNION ALL
            SELECT DISTINCT tags->>'tensorzero::api_key_public_id',
                   NULL,
                   FALSE,
                   1
            FROM tensorzero.json_inferences
            WHERE tags ? 'tensorzero::api_key_public_id'
              AND COALESCE(tags->>'tensorzero::api_key_public_id', '') <> ''
        ) keys
        ORDER BY public_id, rank, description NULLS LAST
        LIMIT $1
        ",
    )
    .bind(API_KEY_SELECT_LIMIT)
    .fetch_all(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to list inference API keys: {e}"),
        })
    })?;

    Ok(Json(ListInferenceApiKeysResponse { api_keys }))
}

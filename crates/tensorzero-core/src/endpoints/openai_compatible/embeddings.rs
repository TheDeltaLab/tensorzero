// Modified by Delta-AI under Apache 2.0
//! Embeddings endpoint handler for OpenAI-compatible API.
//!
//! This module implements the HTTP handler for the `/openai/v1/embeddings` and
//! `/v1/embeddings` endpoints.

use axum::Extension;
use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};

use crate::endpoints::embeddings::{EmbeddingsParams, embeddings};
use crate::routing::RoutingSession;
use crate::utils::gateway::{AppState, AppStateData};
use tensorzero_auth::middleware::RequestApiKeyExtension;

use super::infer::error_response;
use super::synapse::{
    SynapseRequestContext, resolve_cache_options, resolve_openai_compatible_model,
    run_with_request_timeout, served_by_from_model_name,
};
use super::types::embeddings::{OpenAICompatibleEmbeddingParams, OpenAIEmbeddingResponse};
use super::{OpenAICompatibleError, OpenAIStructuredJson};

pub async fn embeddings_handler(
    State(AppStateData {
        config,
        http_client,
        clickhouse_connection_info,
        postgres_connection_info,
        cache_manager,
        deferred_tasks,
        rate_limiting_manager,
        ..
    }): AppState,
    api_key_ext: Option<Extension<RequestApiKeyExtension>>,
    headers: HeaderMap,
    OpenAIStructuredJson(mut openai_compatible_params): OpenAIStructuredJson<
        OpenAICompatibleEmbeddingParams,
    >,
) -> Result<Response, OpenAICompatibleError> {
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
    openai_compatible_params.model = match resolve_openai_compatible_model(
        &openai_compatible_params.model,
        synapse.provider.as_deref(),
    ) {
        Ok(model) => model,
        Err(error) => return Ok(error_response(error, false, &synapse)),
    };
    synapse.served_by = Some(served_by_from_model_name(&openai_compatible_params.model));

    let include_raw_response = openai_compatible_params.tensorzero_include_raw_response;
    let explicit_cache = openai_compatible_params.tensorzero_cache_options.clone();
    let mut embedding_params: EmbeddingsParams = match openai_compatible_params.try_into() {
        Ok(params) => params,
        Err(error) => return Ok(error_response(error, include_raw_response, &synapse)),
    };
    embedding_params.cache_options = resolve_cache_options(explicit_cache, synapse.cache_disabled);
    let session = RoutingSession::new(synapse.fallback_disabled);
    let mut response = match Box::pin(run_with_request_timeout(
        synapse.request_timeout,
        RoutingSession::scope(session.clone(), async {
            embeddings(
                config,
                &http_client,
                clickhouse_connection_info,
                postgres_connection_info,
                cache_manager,
                deferred_tasks,
                rate_limiting_manager,
                embedding_params,
                api_key_ext,
            )
            .await
        }),
    ))
    .await
    {
        Ok(response) => Json(OpenAIEmbeddingResponse::from(response)).into_response(),
        Err(e) => e.into_response_with_raw_entries(true, include_raw_response),
    };
    if let Some(outcome) = session.take_outcome() {
        synapse.served_by = Some(outcome.served_by);
        synapse.fallback_count = outcome.fallback_count;
    }
    synapse.apply_to_response(&mut response);
    Ok(response)
}

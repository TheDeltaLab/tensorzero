// Modified by Delta-AI under Apache 2.0
//! Shared inference execution for OpenAI-compatible chat / completions / responses.

use axum::Extension;
use axum::http::HeaderMap;
use axum::response::Response;

use crate::endpoints::inference::{InferenceOutput, Params, inference};
use crate::error::{Error, ErrorDetails};
use crate::routing::RoutingSession;
use crate::utils::gateway::AppStateData;
use tensorzero_auth::middleware::RequestApiKeyExtension;

use super::synapse::{
    SynapseRequestContext, apply_compat_to_params, resolve_cache_options,
    resolve_openai_compatible_model, run_with_request_timeout,
};
use super::types::chat_completions::OpenAICompatibleParams;

pub struct OpenAICompatibleInference {
    pub output: InferenceOutput,
    pub synapse: SynapseRequestContext,
    pub response_model_prefix: String,
    pub include_usage: bool,
    pub include_raw_usage: bool,
    pub include_original_response: bool,
    pub include_raw_response: bool,
}

/// Validate an OpenAI-compatible chat-style body, apply Synapse headers, and
/// run inference. HTTP-layer errors are returned as already-formatted responses
/// (with Synapse headers) so callers can `return Ok(response)`.
pub async fn infer_openai_compatible(
    state: &AppStateData,
    api_key_ext: Option<Extension<RequestApiKeyExtension>>,
    headers: &HeaderMap,
    mut openai_compatible_params: OpenAICompatibleParams,
) -> Result<OpenAICompatibleInference, Response> {
    let mut synapse = match SynapseRequestContext::try_from_headers(headers) {
        Ok(ctx) => ctx,
        Err(error) => {
            return Err(error_response(
                error,
                false,
                &SynapseRequestContext::from_headers(headers),
            ));
        }
    };

    openai_compatible_params.model = match resolve_openai_compatible_model(
        &openai_compatible_params.model,
        synapse.provider.as_deref(),
    ) {
        Ok(model) => model,
        Err(error) => return Err(error_response(error, false, &synapse)),
    };

    if let Some(n) = openai_compatible_params.n
        && n != 1
    {
        return Err(error_response(
            Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                message: "TensorZero does not support `n` other than 1. Please omit this parameter or set it to 1.".to_string(),
            }),
            false,
            &synapse,
        ));
    }

    if !openai_compatible_params.unknown_fields.is_empty() {
        if openai_compatible_params.tensorzero_deny_unknown_fields {
            let mut unknown_field_names = openai_compatible_params
                .unknown_fields
                .keys()
                .cloned()
                .collect::<Vec<_>>();

            unknown_field_names.sort();
            let unknown_field_names = unknown_field_names.join(", ");

            return Err(error_response(
                Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                    message: format!(
                        "`tensorzero::deny_unknown_fields` is set to true, but found unknown fields in the request: [{unknown_field_names}]"
                    ),
                }),
                false,
                &synapse,
            ));
        }
        tracing::warn!(
            "Ignoring unknown fields in OpenAI-compatible request: {:?}",
            openai_compatible_params
                .unknown_fields
                .keys()
                .collect::<Vec<_>>()
        );
    }

    let include_raw_usage = openai_compatible_params.tensorzero_include_raw_usage;
    let include_original_response = openai_compatible_params.tensorzero_include_original_response;
    let include_raw_response = openai_compatible_params.tensorzero_include_raw_response;

    if include_original_response {
        tracing::warn!(
            "The `tensorzero::include_original_response` parameter is deprecated. Use `tensorzero::include_raw_response` instead."
        );
    }

    let explicit_include_usage = openai_compatible_params
        .stream_options
        .as_ref()
        .map(|opts| opts.include_usage);

    if openai_compatible_params.stream.unwrap_or(false)
        && include_raw_usage
        && explicit_include_usage == Some(false)
    {
        return Err(error_response(
            Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                message: "`tensorzero::include_raw_usage` requires `stream_options.include_usage` to be true (or omitted) for streaming requests".to_string(),
            }),
            include_raw_response,
            &synapse,
        ));
    }

    let include_usage = explicit_include_usage.unwrap_or(false) || include_raw_usage;

    let explicit_cache = openai_compatible_params.tensorzero_cache_options.clone();
    let mut params = match Params::try_from_openai(openai_compatible_params) {
        Ok(params) => params,
        Err(error) => return Err(error_response(error, include_raw_response, &synapse)),
    };
    if let Err(error) = apply_compat_to_params(headers, &mut params) {
        return Err(error_response(error, include_raw_response, &synapse));
    }
    params.cache_options = resolve_cache_options(explicit_cache, synapse.cache_disabled);
    params.extra_internal_tags = synapse.observability_tags(headers);

    synapse = synapse.with_served_by_from_params(&params);

    let response_model_prefix = match (&params.function_name, &params.model_name) {
        (Some(function_name), None) => {
            format!("tensorzero::function_name::{function_name}::variant_name::")
        }
        (None, Some(_model_name)) => "tensorzero::model_name::".to_string(),
        (Some(_), Some(_)) => {
            return Err(error_response(
                ErrorDetails::InvalidInferenceTarget {
                    message: "Only one of `function_name` or `model_name` can be provided"
                        .to_string(),
                }
                .into(),
                include_raw_response,
                &synapse,
            ));
        }
        (None, None) => {
            return Err(error_response(
                ErrorDetails::InvalidInferenceTarget {
                    message: "Either `function_name` or `model_name` must be provided".to_string(),
                }
                .into(),
                include_raw_response,
                &synapse,
            ));
        }
    };

    let session = RoutingSession::new(synapse.fallback_disabled);
    let inference_result = Box::pin(run_with_request_timeout(
        synapse.request_timeout,
        RoutingSession::scope(session.clone(), async {
            Box::pin(inference(
                state.config.clone(),
                &state.http_client,
                state.clickhouse_connection_info.clone(),
                state.postgres_connection_info.clone(),
                state.cache_manager.clone(),
                state.deferred_tasks.clone(),
                state.rate_limiting_manager.clone(),
                state.primary_datastore,
                params,
                api_key_ext,
            ))
            .await
        }),
    ))
    .await;

    if let Some(outcome) = session.take_outcome() {
        synapse.served_by = Some(outcome.served_by);
        synapse.fallback_count = outcome.fallback_count;
    }

    let output = match inference_result {
        Ok(data) => data.output,
        Err(error) => {
            return Err(error_response(error, include_raw_response, &synapse));
        }
    };

    Ok(OpenAICompatibleInference {
        output,
        synapse,
        response_model_prefix,
        include_usage,
        include_raw_usage,
        include_original_response,
        include_raw_response,
    })
}

pub(super) fn error_response(
    error: Error,
    include_raw_response: bool,
    synapse: &SynapseRequestContext,
) -> Response {
    let mut response = error.into_response_with_raw_entries(true, include_raw_response);
    synapse.apply_to_response(&mut response);
    response
}

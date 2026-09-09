// Modified by Delta-AI under Apache 2.0
use futures::StreamExt;
use futures::future::try_join_all;
use indexmap::IndexMap;
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;
use tensorzero_stored_config::{StoredCostConfig, StoredTimeoutsConfig, StoredUnifiedCostConfig};
use tensorzero_stored_config::{StoredModelConfig, StoredModelProvider, StoredProviderConfig};
use tokio::time::error::Elapsed;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{Level, Span, span};
use tracing_futures::{Instrument, Instrumented};
use tracing_opentelemetry::OpenTelemetrySpanExt;
use url::Url;
use uuid::Uuid;

use crate::cache::{
    CacheData, CacheValidationInfo, ModelProviderRequest, NonStreamingCacheData,
    StreamingCacheData, cache_lookup, cache_lookup_streaming, start_cache_write,
    start_cache_write_streaming,
};
use crate::config::with_skip_credential_validation;
use crate::config::{
    Namespace, OtlpConfig, OtlpTracesFormat, TimeoutsConfig, provider_types::ProviderTypesConfig,
};
use crate::cost::{
    CostConfig, ResponseMode, apply_computed_cost, load_cost_config_with_provider_defaults,
    load_unified_cost_config_with_provider_defaults,
};
use crate::db::delegating_connection::DelegatingDatabaseConnection;
use crate::db::model_inferences::ModelInferenceQueries;
use crate::endpoints::inference::InferenceClients;
use crate::http::TensorzeroHttpClient;
use crate::inference::types::StoredModelInference;
use crate::inference::types::usage::aggregate_usage_from_single_streaming_model_inference;
use crate::model_table::ProviderKind;
use crate::observability::genai_conventions;
use crate::observability::internal_metrics::{
    TENSORZERO_INPUT_TOKENS_TOTAL, TENSORZERO_OUTPUT_TOKENS_TOTAL,
};
use crate::observability::openinference_conventions;
#[cfg(any(test, feature = "e2e_tests"))]
use crate::providers::dummy::DummyProvider;
use crate::providers::google_ai_studio_gemini::GoogleAIStudioGeminiProvider;
use tensorzero_types::{UninitializedCostConfig, UninitializedUnifiedCostConfig};

use crate::inference::types::ProviderInferenceResponseExt;
use crate::inference::types::batch::{
    BatchRequestRow, PollBatchInferenceResponse, StartBatchModelInferenceResponse,
    StartBatchProviderInferenceResponse,
};
use crate::inference::types::extra_body::ExtraBodyConfig;
use crate::inference::types::extra_headers::ExtraHeadersConfig;
use crate::inference::types::{
    ApiType, ContentBlock, PeekableProviderInferenceResponseStream, ProviderInferenceResponseChunk,
    ProviderInferenceResponseStreamInner, RawResponseEntry, RequestMessage, Thought, Unknown,
    Usage, stream_with_deadline,
};
use crate::model_table::{
    AnthropicKind, BaseModelTable, DeepSeekKind, FireworksKind, GoogleAIStudioGeminiKind, GroqKind,
    HyperbolicKind, MistralKind, OpenAIKind, OpenRouterKind, ProviderTypeDefaultCredentials,
    ShorthandModelConfig, TogetherKind, XAIKind,
};
use crate::providers::helpers::peek_first_chunk;
use crate::providers::hyperbolic::HyperbolicProvider;
use crate::providers::openai::OpenAIAPIType;
use crate::rate_limiting::{RateLimitResourceUsage, TicketBorrows, decimal_cost_to_nano_cost};
use crate::{
    endpoints::inference::InferenceCredentials,
    error::{Error, ErrorDetails, TimeoutKind},
    inference::types::{ModelInferenceRequest, ModelInferenceResponse, ProviderInferenceResponse},
};
use metrics::counter;
use serde::{Deserialize, Serialize};
use tensorzero_stored_config::{StoredExtraBodyConfig, StoredExtraHeadersConfig};

use crate::providers::{
    anthropic::AnthropicProvider, deepseek::DeepSeekProvider, fireworks::FireworksProvider,
    gcp_vertex_anthropic::GCPVertexAnthropicProvider, gcp_vertex_gemini::GCPVertexGeminiProvider,
    groq::GroqProvider, mistral::MistralProvider, openai::OpenAIProvider,
    openrouter::OpenRouterProvider, together::TogetherProvider, xai::XAIProvider,
};

pub(crate) fn record_usage_metrics(usage: &Usage) {
    if let Some(input_tokens) = usage.input_tokens {
        counter!("tensorzero_input_tokens_total").increment(input_tokens as u64);
        TENSORZERO_INPUT_TOKENS_TOTAL.fetch_add(input_tokens as u64, Ordering::Relaxed);
    }
    if let Some(output_tokens) = usage.output_tokens {
        counter!("tensorzero_output_tokens_total").increment(output_tokens as u64);
        TENSORZERO_OUTPUT_TOKENS_TOTAL.fetch_add(output_tokens as u64, Ordering::Relaxed);
    }
}

#[derive(ts_rs::TS, Debug, Serialize)]
#[ts(export)]
pub struct ModelConfig {
    pub routing: Vec<Arc<str>>, // [provider name A, provider name B, ...]
    pub providers: HashMap<Arc<str>, ModelProvider>, // provider name => provider config
    pub timeouts: TimeoutsConfig,
    pub skip_relay: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub namespace: Option<Namespace>,
}

#[derive(ts_rs::TS, Clone, Debug, Deserialize, PartialEq, Serialize)]
#[ts(export, optional_fields)]
#[serde(deny_unknown_fields)]
pub struct UninitializedModelConfig {
    pub routing: Vec<Arc<str>>, // [provider name A, provider name B, ...]
    pub providers: HashMap<Arc<str>, UninitializedModelProvider>, // provider name => provider config
    #[serde(default)]
    pub timeouts: TimeoutsConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skip_relay: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<Namespace>,
}

impl UninitializedModelConfig {
    pub async fn load(
        self,
        model_name: &str,
        provider_types: &ProviderTypesConfig,
        provider_type_default_credentials: &ProviderTypeDefaultCredentials,
        relay_mode: bool,
        is_config_snapshot: bool,
    ) -> Result<ModelConfig, Error> {
        let skip_relay = self.skip_relay.unwrap_or(false);
        // We want `ModelProvider` to know its own name (from the 'providers' config section).
        // We first deserialize to `HashMap<Arc<str>, UninitializedModelProvider>`, and then
        // build `ModelProvider`s using the name keys from the map.
        let providers = try_join_all(self.providers.into_iter().map(|(name, provider)| {
            async move {
                let load_future = provider.config.load(
                    provider_types,
                    provider_type_default_credentials,
                    is_config_snapshot,
                );

                // In relay mode, don't run credential validation for providers,
                // since requests to the parent model get redirected to the downstream gateway.
                // The exception is `skip_relay` models - we'll still use their providers,
                // so we want to run credential validation for them.
                let config = if relay_mode && !skip_relay {
                    Box::pin(with_skip_credential_validation(load_future)).await
                } else {
                    load_future.await
                };
                let default_timezone = provider.timezone.clone();
                let currency = provider.currency.as_deref();
                let cost = provider
                    .cost
                    .map(|c| {
                        load_cost_config_with_provider_defaults(
                            c,
                            default_timezone.as_deref(),
                            currency,
                        )
                    })
                    .transpose()
                    .map_err(|e| {
                        Error::new(ErrorDetails::Config {
                            message: format!("models.{model_name}.providers.{name}.cost: {e}"),
                        })
                    })?;
                let batch_cost = provider
                    .batch_cost
                    .map(|c| {
                        load_unified_cost_config_with_provider_defaults(
                            c,
                            default_timezone.as_deref(),
                            currency,
                        )
                    })
                    .transpose()
                    .map_err(|e| {
                        Error::new(ErrorDetails::Config {
                            message: format!(
                                "models.{model_name}.providers.{name}.batch_cost: {e}"
                            ),
                        })
                    })?;
                Ok::<_, Error>((
                    name.clone(),
                    ModelProvider {
                        name: name.clone(),
                        config: config.map_err(|e| {
                            Error::new(ErrorDetails::Config {
                                message: format!("models.{model_name}.providers.{name}: {e}"),
                            })
                        })?,
                        extra_body: provider.extra_body,
                        extra_headers: provider.extra_headers,
                        timeouts: provider.timeouts,
                        discard_unknown_chunks: provider.discard_unknown_chunks,
                        cost,
                        batch_cost,
                    },
                ))
            }
        }))
        .await?
        .into_iter()
        .collect::<HashMap<_, _>>();
        Ok(ModelConfig {
            routing: self.routing,
            providers,
            timeouts: self.timeouts,
            skip_relay,
            namespace: self.namespace,
        })
    }
}

impl TryFrom<StoredModelConfig> for UninitializedModelConfig {
    type Error = Error;

    fn try_from(stored: StoredModelConfig) -> Result<Self, Error> {
        let providers = stored
            .providers
            .into_iter()
            .map(|(name, provider)| {
                let provider: UninitializedModelProvider = provider.try_into()?;
                Ok((Arc::<str>::from(name), provider))
            })
            .collect::<Result<HashMap<_, _>, Error>>()?;
        Ok(UninitializedModelConfig {
            routing: stored.routing.into_iter().map(Arc::<str>::from).collect(),
            providers,
            timeouts: stored
                .timeouts
                .map(TimeoutsConfig::from)
                .unwrap_or_default(),
            skip_relay: stored.skip_relay,
            namespace: stored.namespace.map(Namespace::new).transpose()?,
        })
    }
}

impl TryFrom<StoredModelProvider> for UninitializedModelProvider {
    type Error = Error;

    fn try_from(stored: StoredModelProvider) -> Result<Self, Error> {
        let config: UninitializedProviderConfig = stored.provider.try_into()?;
        let cost = stored.cost.map(UninitializedCostConfig::from);
        let batch_cost = stored.batch_cost.map(UninitializedUnifiedCostConfig::from);
        Ok(UninitializedModelProvider {
            config,
            extra_body: stored.extra_body.map(ExtraBodyConfig::from),
            extra_headers: stored.extra_headers.map(ExtraHeadersConfig::from),
            timeouts: stored
                .timeouts
                .map(TimeoutsConfig::from)
                .unwrap_or_default(),
            discard_unknown_chunks: stored.discard_unknown_chunks.unwrap_or_default(),
            cost,
            batch_cost,
            timezone: stored.timezone,
            currency: stored.currency,
        })
    }
}

impl TryFrom<&UninitializedModelConfig> for StoredModelConfig {
    type Error = Error;

    fn try_from(config: &UninitializedModelConfig) -> Result<Self, Error> {
        let providers = config
            .providers
            .iter()
            .map(|(provider_name, provider)| {
                Ok::<_, Error>((
                    provider_name.to_string(),
                    StoredModelProvider::from(provider),
                ))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;

        Ok(StoredModelConfig {
            routing: config.routing.iter().map(ToString::to_string).collect(),
            providers,
            timeouts: Some(StoredTimeoutsConfig::from(&config.timeouts)),
            skip_relay: config.skip_relay,
            namespace: config.namespace.as_ref().map(ToString::to_string),
        })
    }
}

// Modified by Delta-AI under Apache 2.0
pub use tensorzero_providers::provider_config::{
    HostedProviderKind, ProviderConfig, UninitializedProviderConfig,
};

pub struct StreamResponse {
    pub stream: Instrumented<PeekableProviderInferenceResponseStream>,
    pub raw_request: String,
    pub model_provider_name: Arc<str>,
    pub provider_type: Arc<str>,
    pub cached: bool,
    pub model_inference_id: Uuid,
    /// Raw response entries from failed provider attempts during fallback.
    pub failed_raw_response: Vec<RawResponseEntry>,
    /// Cost configuration from the successful provider, for computing cost after streaming completes.
    pub cost_config: Option<CostConfig>,
}

impl StreamResponse {
    pub fn from_cache(
        cache_lookup: CacheData<StreamingCacheData>,
        model_provider_name: Arc<str>,
        provider_type: Arc<str>,
        model_inference_id: Uuid,
    ) -> Self {
        let chunks = cache_lookup.output.chunks;
        let chunks_len = chunks.len();

        Self {
            stream: (Box::pin(futures::stream::iter(chunks.into_iter().enumerate().map(
                move |(index, c)| {
                    Ok(ProviderInferenceResponseChunk {
                        content: c.content,
                        raw_response: c.raw_response,
                        // We intentionally don't cache and re-use these values from the original
                        // request:
                        // Use the real usage (so that the `ModelInference` row we write is accurate)
                        // The usage returned to over HTTP is adjusted in `InferenceResponseChunk::new`
                        usage: c.usage,
                        // raw_usage is not cached
                        raw_usage: None,
                        // We didn't make any network calls to the model provider, so the latency is 0
                        provider_latency: Duration::from_secs(0),
                        // For all chunks but the last one, the finish reason is None
                        // For the last chunk, the finish reason is the same as the cache lookup
                        finish_reason: if index == chunks_len - 1 {
                            cache_lookup.finish_reason
                        } else {
                            None
                        },
                    })
                },
            ))) as ProviderInferenceResponseStreamInner)
                .peekable()
                .instrument(tracing::info_span!(
                    "stream_from_cache",
                    otel.name = "stream_from_cache"
                )),
            raw_request: cache_lookup.raw_request,
            model_provider_name,
            provider_type,
            cached: true,
            model_inference_id,
            failed_raw_response: vec![],
            cost_config: None,
        }
    }
}

/// Records a failed provider attempt as a `ModelInference` row with an `error`
/// column (fire-and-forget via `deferred_tasks`). No-op unless the caller opted
/// in through `InferenceClients::failed_model_inference_datastore` (gated by
/// `dryrun`, `observability.enabled`, and `record_failed_inferences` at the
/// endpoint layer). Write failures are logged and swallowed.
fn record_failed_model_inference(
    clients: &InferenceClients,
    inference_id: Uuid,
    model_name: &str,
    provider_name: &str,
    function_name: Option<&str>,
    error: &Error,
    response_time: Duration,
) {
    let Some(primary_datastore) = clients.failed_model_inference_datastore else {
        return;
    };
    let row = StoredModelInference::failed(
        inference_id,
        function_name.unwrap_or_default().to_string(),
        // The variant name is not visible at the model layer
        String::new(),
        model_name.to_string(),
        provider_name.to_string(),
        error,
        Some(response_time),
        None,
    );
    let clickhouse_connection_info = clients.clickhouse_connection_info.clone();
    let postgres_connection_info = clients.postgres_connection_info.clone();
    clients.deferred_tasks.spawn(async move {
        let database = DelegatingDatabaseConnection::new(
            clickhouse_connection_info,
            postgres_connection_info,
            primary_datastore,
        );
        if let Err(e) = database.insert_model_inferences(&[row]).await {
            tracing::warn!("Failed to write failed model inference to the database: {e}");
        }
    });
}

impl ModelConfig {
    /// Checks if an Unknown content block should be filtered out based on model_name and provider_name.
    /// Returns true if the block should be filtered (removed), false if it should be kept.
    fn should_filter_unknown_block(
        block_model_name: Option<&String>,
        block_provider_name: Option<&String>,
        target_model_name: &str,
        target_provider_name: &str,
    ) -> bool {
        // If model_name is specified and doesn't match, filter it out
        if let Some(m) = block_model_name
            && m != target_model_name
        {
            return true;
        }
        // If provider_name is specified and doesn't match, filter it out
        if let Some(p) = block_provider_name
            && p != target_provider_name
        {
            return true;
        }
        // Keep the block if both match (or are None)
        false
    }

    fn filter_content_blocks<'a>(
        request: &'a ModelInferenceRequest<'a>,
        model_name: &str,
        provider: &ModelProvider,
    ) -> Cow<'a, ModelInferenceRequest<'a>> {
        let provider_name = provider.name.as_ref();
        let needs_filter = request.messages.iter().any(|m| {
            m.content.iter().any(|c| match c {
                ContentBlock::Unknown(Unknown {
                    model_name: block_model_name,
                    provider_name: block_provider_name,
                    data: _,
                }) => Self::should_filter_unknown_block(
                    block_model_name.as_ref(),
                    block_provider_name.as_ref(),
                    model_name,
                    provider_name,
                ),
                ContentBlock::Thought(Thought {
                    text: _,
                    signature: _,
                    summary: _,
                    provider_type,
                    extra_data: _,
                }) => provider_type
                    .as_ref()
                    .is_some_and(|t| t != &provider.config.thought_block_provider_type()),
                _ => false,
            })
        });
        if needs_filter {
            let new_messages = request
                .messages
                .iter()
                .map(|m| RequestMessage {
                    content: m
                        .content
                        .iter()
                        .flat_map(|c| match c {
                            ContentBlock::Unknown(Unknown {
                                model_name: block_model_name,
                                provider_name: block_provider_name,
                                data: _,
                            }) => {
                                if Self::should_filter_unknown_block(
                                    block_model_name.as_ref(),
                                    block_provider_name.as_ref(),
                                    model_name,
                                    provider_name,
                                ) {
                                    None
                                } else {
                                    Some(c.clone())
                                }
                            }
                            ContentBlock::Thought(Thought {
                                text: _,
                                signature: _,
                                summary: _,
                                provider_type,
                                extra_data: _,
                            }) => {
                                // When a thought is scoped to a particular provider type, we discard
                                // if it doesn't match our target provider.
                                // Thoughts without a `provider_type` are used for all providers.
                                if provider_type.as_ref().is_some_and(|t| {
                                    t != &provider.config.thought_block_provider_type()
                                }) {
                                    None
                                } else {
                                    Some(c.clone())
                                }
                            }
                            _ => Some(c.clone()),
                        })
                        .collect(),
                    ..m.clone()
                })
                .collect();
            Cow::Owned(ModelInferenceRequest {
                messages: new_messages,
                ..request.clone()
            })
        } else {
            Cow::Borrowed(request)
        }
    }

    /// Performs a non-streaming request to a specific provider, performing a cache lookup if enabled.
    /// We apply model-provider timeouts to the future produced by this function
    /// (as we want to apply the timeout to ClickHouse cache lookups)
    async fn non_streaming_provider_request<'request>(
        &self,
        model_provider_request: ModelProviderRequest<'request>,
        provider: &'request ModelProvider,
        clients: &InferenceClients,
    ) -> Result<ModelInferenceResponse, Error> {
        let provider_type: Arc<str> = Arc::from(provider.provider_type());
        // TODO: think about how to best handle errors here
        if clients.cache_options.enabled.read() {
            let cache_lookup = cache_lookup(
                &clients.cache_manager,
                model_provider_request,
                clients.cache_options.max_age_s,
                provider_type.clone(),
            )
            .await
            .ok()
            .flatten();
            if let Some(cache_lookup) = cache_lookup {
                return Ok(cache_lookup);
            }
        }
        let response = provider
            .infer(model_provider_request, clients)
            .instrument(span!(
                Level::INFO,
                "infer",
                provider_name = model_provider_request.provider_name
            ))
            .await?;
        // We already checked the cache above (and returned early if it was a hit), so this response was not from the cache
        Ok(ModelInferenceResponse::new(
            response,
            model_provider_request.provider_name.into(),
            provider_type,
            false,
        ))
    }

    /// Performs a streaming request to a specific provider, performing a cache lookup if enabled.
    /// We apply model-provider timeouts to the future produced by this function
    /// (as we want to apply the timeout to ClickHouse cache lookups).
    ///
    /// This function also includes a call to `peek_first_chunk` - this ensure that the
    /// duration of the returned future includes the time taken to get the first chunk
    /// from the model provider.
    async fn streaming_provider_request<'request>(
        &self,
        model_provider_request: ModelProviderRequest<'request>,
        provider: &'request ModelProvider,
        clients: &InferenceClients,
    ) -> Result<StreamResponseAndMessages, Error> {
        let provider_type: Arc<str> = Arc::from(provider.provider_type());
        // TODO: think about how to best handle errors here
        if clients.cache_options.enabled.read() {
            let cache_lookup = cache_lookup_streaming(
                &clients.cache_manager,
                model_provider_request,
                clients.cache_options.max_age_s,
                provider_type.clone(),
            )
            .await
            .ok()
            .flatten();
            if let Some(cache_lookup) = cache_lookup {
                return Ok(StreamResponseAndMessages {
                    response: cache_lookup,
                    messages: model_provider_request.request.messages.clone(),
                });
            }
        }

        let StreamAndRawRequest {
            stream,
            raw_request,
            ticket_borrow,
        } = provider
            .infer_stream(model_provider_request, clients)
            .await?;

        // Note - we cache the chunks here so that we store the raw model provider input and response chunks
        // in the cache. We don't want this logic in `collect_chunks`, which would cause us to cache the result
        // of higher-level transformations (e.g. dicl)
        let write_to_cache = clients.cache_options.enabled.write();
        let span = stream.span().clone();
        let mut stream = wrap_provider_stream(
            raw_request.clone(),
            model_provider_request,
            ticket_borrow,
            clients,
            stream,
            write_to_cache,
        )?
        .instrument(span);
        // Get a single chunk from the stream and make sure it is OK then send to client.
        // We want to do this here so that we can tell that the request is working.
        peek_first_chunk(
            stream.inner_mut(),
            &raw_request,
            model_provider_request.provider_name,
            provider.api_type(),
        )
        .await?;
        Ok(StreamResponseAndMessages {
            response: StreamResponse {
                stream,
                raw_request,
                model_provider_name: model_provider_request.provider_name.into(),
                provider_type,
                cached: false,
                model_inference_id: model_provider_request.model_inference_id,
                failed_raw_response: vec![],
                cost_config: provider.cost.clone(),
            },
            messages: model_provider_request.request.messages.clone(),
        })
    }

    #[tracing::instrument(skip_all, fields(model_name = model_name, otel.name = "model_inference", stream = false))]
    pub async fn infer<'request>(
        &self,
        request: &'request ModelInferenceRequest<'request>,
        clients: &InferenceClients,
        model_name: &'request str,
        function_name: Option<&'request str>,
    ) -> Result<ModelInferenceResponse, Error> {
        let span = tracing::Span::current();
        clients.otlp_config.mark_openinference_chain_span(&span);

        let mut provider_errors: IndexMap<String, Error> = IndexMap::new();
        let run_all_models = async {
            if let Some(relay) = &clients.relay
                && !self.skip_relay
            {
                let response = relay
                    .relay_non_streaming(model_name, request, clients)
                    .await?;
                return Ok(ModelInferenceResponse::new(
                    response,
                    "tensorzero::relay".into(),
                    "tensorzero::relay".into(),
                    false,
                ));
            }
            for provider_name in &crate::routing::effective_routing(&self.routing) {
                let provider = self.providers.get(provider_name).ok_or_else(|| {
                    Error::new(ErrorDetails::ProviderNotFound {
                        provider_name: provider_name.to_string(),
                    })
                })?;
                let request = Self::filter_content_blocks(request, model_name, provider);
                let model_provider_request = ModelProviderRequest {
                    request: &request,
                    model_name,
                    provider_name,
                    otlp_config: &clients.otlp_config,
                    model_inference_id: Uuid::now_v7(),
                    function_name,
                };
                let cache_key = model_provider_request.get_cache_key()?;

                let response_fut =
                    self.non_streaming_provider_request(model_provider_request, provider, clients);
                let attempt_start = tokio::time::Instant::now();
                let response = if let Some(timeout) = provider.non_streaming_total_timeout() {
                    tokio::time::timeout(timeout, response_fut)
                        .await
                        // Convert the outer `Elapsed` error into a TensorZero error,
                        // so that it can be handled by the `match response` block below
                        .unwrap_or_else(|_: Elapsed| {
                            Err(Error::new(ErrorDetails::ModelProviderTimeout {
                                provider_name: provider_name.to_string(),
                                timeout,
                                kind: TimeoutKind::NonStreamingTotal,
                            }))
                        })
                } else {
                    response_fut.await
                };

                match response {
                    Ok(mut response) => {
                        // Compute cost from raw response using provider's cost config
                        if !response.cached
                            && let Some(cost_config) = &provider.cost
                        {
                            apply_computed_cost(
                                &mut response.usage,
                                &response.raw_response,
                                cost_config,
                                ResponseMode::NonStreaming,
                            );
                        }

                        // Perform the cache write outside of the `non_streaming_total_timeout` timeout future,
                        // (in case we ever add a blocking cache write option)
                        if !response.cached && clients.cache_options.enabled.write() {
                            let _ = start_cache_write(
                                &clients.cache_manager,
                                cache_key,
                                CacheData {
                                    output: NonStreamingCacheData {
                                        blocks: response.output.clone(),
                                    },
                                    raw_request: response.raw_request.clone(),
                                    raw_response: response.raw_response.clone(),
                                    input_tokens: response.usage.input_tokens,
                                    output_tokens: response.usage.output_tokens,
                                    finish_reason: response.finish_reason,
                                },
                                CacheValidationInfo {
                                    tool_config: request
                                        .tool_config
                                        .clone()
                                        .map(std::borrow::Cow::into_owned),
                                },
                            );
                        }

                        // Collect raw response entries from failed providers for fallback reporting
                        if clients.include_raw_response {
                            for error in provider_errors.values() {
                                if let Some(entries) = error.extract_raw_response() {
                                    response.failed_raw_response.extend(entries);
                                }
                            }
                        }

                        if let Some(session) = crate::routing::RoutingSession::current() {
                            session.record_provider(&self.routing, provider_name, model_name);
                        }
                        if !response.cached {
                            crate::routing::record_tokens_per_sec_from_latency(
                                provider_name,
                                model_name,
                                &response.usage,
                                &response.provider_latency,
                            );
                        }

                        return Ok(response);
                    }
                    Err(error) => {
                        if let Some(session) = crate::routing::RoutingSession::current() {
                            session.record_provider(&self.routing, provider_name, model_name);
                        }
                        record_failed_model_inference(
                            clients,
                            request.inference_id,
                            model_name,
                            provider_name,
                            function_name,
                            &error,
                            attempt_start.elapsed(),
                        );
                        if !crate::routing::should_failover(&error) {
                            return Err(error);
                        }
                        provider_errors.insert(provider_name.to_string(), error);
                    }
                }
            }
            Err(Error::new(ErrorDetails::AllModelProvidersFailed {
                provider_errors,
            }))
        };
        // This is the top-level model timeout, which limits the total time taken to run all providers.
        // Some of the providers may themselves have timeouts, which is fine. Provider timeouts
        // are treated as just another kind of provider error - a timeout of N ms is equivalent
        // to a provider taking N ms, and then producing a normal HTTP error.
        if let Some(timeout) = self
            .timeouts
            .non_streaming
            .as_ref()
            .and_then(|ns| ns.total_ms)
        {
            let timeout = Duration::from_millis(timeout);
            tokio::time::timeout(timeout, run_all_models)
                .await
                // Convert the outer `Elapsed` error into a TensorZero error,
                // so that it can be handled by the `match response` block below
                .unwrap_or_else(|_: Elapsed| {
                    Err(Error::new(ErrorDetails::ModelTimeout {
                        model_name: model_name.to_string(),
                        timeout,
                        kind: TimeoutKind::NonStreamingTotal,
                    }))
                })
        } else {
            run_all_models.await
        }
    }

    #[tracing::instrument(skip_all, fields(model_name = model_name, otel.name = "model_inference", stream = true))]
    pub async fn infer_stream<'request>(
        &self,
        request: &'request ModelInferenceRequest<'request>,
        clients: &InferenceClients,
        model_name: &'request str,
        function_name: Option<&'request str>,
    ) -> Result<StreamResponseAndMessages, Error> {
        clients
            .otlp_config
            .mark_openinference_chain_span(&tracing::Span::current());
        let mut provider_errors: IndexMap<String, Error> = IndexMap::new();
        let run_all_models = async {
            if let Some(relay) = &clients.relay
                && !self.skip_relay
            {
                // Note - we do *not* call wrap_provider_stream,
                // since we don't want caching or (model provider) OTEL attributes
                let (stream, raw_request) =
                    relay.relay_streaming(model_name, request, clients).await?;
                return Ok(StreamResponseAndMessages {
                    response: StreamResponse {
                        stream: stream.instrument(Span::current()),
                        raw_request,
                        model_provider_name: "tensorzero::relay".into(),
                        provider_type: "tensorzero::relay".into(),
                        cached: false,
                        model_inference_id: Uuid::now_v7(),
                        failed_raw_response: vec![],
                        cost_config: None,
                    },
                    messages: request.messages.clone(),
                });
            }
            for provider_name in &crate::routing::effective_routing(&self.routing) {
                let provider = self.providers.get(provider_name).ok_or_else(|| {
                    Error::new(ErrorDetails::ProviderNotFound {
                        provider_name: provider_name.to_string(),
                    })
                })?;
                let request = Self::filter_content_blocks(request, model_name, provider);
                let model_provider_request = ModelProviderRequest {
                    request: &request,
                    model_name,
                    provider_name,
                    otlp_config: &clients.otlp_config,
                    model_inference_id: Uuid::now_v7(),
                    function_name,
                };

                // This future includes a call to `peek_first_chunk`, so applying
                // `streaming_ttft_timeout` is correct.
                let start = tokio::time::Instant::now();
                let response_fut =
                    self.streaming_provider_request(model_provider_request, provider, clients);

                // Compute the effective pre-TTFT deadline from both ttft_ms and total_ms.
                // If both are set, use the earlier deadline. The error kind reflects which
                // timeout was responsible.
                let total_timeout = provider.streaming_total_timeout();
                let ttft_timeout = provider.streaming_ttft_timeout();
                let pre_ttft_timeout = match (ttft_timeout, total_timeout) {
                    (Some(ttft), Some(total)) => {
                        if ttft <= total {
                            Some((start + ttft, ttft, TimeoutKind::StreamingTtft))
                        } else {
                            Some((start + total, total, TimeoutKind::StreamingTotal))
                        }
                    }
                    (Some(ttft), None) => Some((start + ttft, ttft, TimeoutKind::StreamingTtft)),
                    (None, Some(total)) => {
                        Some((start + total, total, TimeoutKind::StreamingTotal))
                    }
                    (None, None) => None,
                };

                let response = if let Some((deadline, timeout, kind)) = pre_ttft_timeout {
                    tokio::time::timeout_at(deadline, response_fut)
                        .await
                        .unwrap_or_else(|_: Elapsed| {
                            Err(Error::new(ErrorDetails::ModelProviderTimeout {
                                provider_name: provider_name.to_string(),
                                timeout,
                                kind,
                            }))
                        })
                } else {
                    response_fut.await
                };

                match response {
                    Ok(mut response) => {
                        // Collect raw response entries from failed providers for fallback reporting
                        if clients.include_raw_response {
                            for error in provider_errors.values() {
                                if let Some(entries) = error.extract_raw_response() {
                                    response.response.failed_raw_response.extend(entries);
                                }
                            }
                        }
                        // Wrap the post-TTFT stream with the remaining total deadline
                        if let Some(total_timeout) = total_timeout {
                            let deadline = start + total_timeout;
                            let span = response.response.stream.span().clone();
                            let inner = response.response.stream.into_inner();
                            let provider_name = provider_name.to_string();
                            let wrapped = stream_with_deadline(inner, deadline, move || {
                                Error::new(ErrorDetails::ModelProviderTimeout {
                                    provider_name,
                                    timeout: total_timeout,
                                    kind: TimeoutKind::StreamingTotal,
                                })
                            });
                            response.response.stream = wrapped.peekable().instrument(span);
                        }
                        if let Some(session) = crate::routing::RoutingSession::current() {
                            session.record_provider(&self.routing, provider_name, model_name);
                        }
                        return Ok(response);
                    }
                    Err(error) => {
                        if let Some(session) = crate::routing::RoutingSession::current() {
                            session.record_provider(&self.routing, provider_name, model_name);
                        }
                        record_failed_model_inference(
                            clients,
                            request.inference_id,
                            model_name,
                            provider_name,
                            function_name,
                            &error,
                            start.elapsed(),
                        );
                        if !crate::routing::should_failover(&error) {
                            return Err(error);
                        }
                        provider_errors.insert(provider_name.to_string(), error);
                    }
                }
            }
            Err(Error::new(ErrorDetails::AllModelProvidersFailed {
                provider_errors,
            }))
        };
        // See the corresponding `non_streaming.total_ms` timeout in the `infer`
        // method above for more details.
        let start = tokio::time::Instant::now();

        // Compute the effective pre-TTFT deadline from both ttft_ms and total_ms.
        let ttft_timeout = self
            .timeouts
            .streaming
            .as_ref()
            .and_then(|s| s.ttft_ms)
            .map(Duration::from_millis);
        let streaming_total_ms = self.timeouts.streaming.as_ref().and_then(|s| s.total_ms);
        let total_timeout = streaming_total_ms.map(Duration::from_millis);
        let pre_ttft_timeout = match (ttft_timeout, total_timeout) {
            (Some(ttft), Some(total)) => {
                if ttft <= total {
                    Some((start + ttft, ttft, TimeoutKind::StreamingTtft))
                } else {
                    Some((start + total, total, TimeoutKind::StreamingTotal))
                }
            }
            (Some(ttft), None) => Some((start + ttft, ttft, TimeoutKind::StreamingTtft)),
            (None, Some(total)) => Some((start + total, total, TimeoutKind::StreamingTotal)),
            (None, None) => None,
        };

        let mut result = if let Some((deadline, timeout, kind)) = pre_ttft_timeout {
            tokio::time::timeout_at(deadline, run_all_models)
                .await
                .unwrap_or_else(|_: Elapsed| {
                    Err(Error::new(ErrorDetails::ModelTimeout {
                        model_name: model_name.to_string(),
                        timeout,
                        kind,
                    }))
                })
        } else {
            run_all_models.await
        }?;

        // Wrap the post-TTFT stream with the remaining total deadline
        if let Some(total_ms) = streaming_total_ms {
            let total_timeout = Duration::from_millis(total_ms);
            let deadline = start + total_timeout;
            let span = result.response.stream.span().clone();
            let inner = result.response.stream.into_inner();
            let model_name = model_name.to_string();
            let wrapped = stream_with_deadline(inner, deadline, move || {
                Error::new(ErrorDetails::ModelTimeout {
                    model_name,
                    timeout: total_timeout,
                    kind: TimeoutKind::StreamingTotal,
                })
            });
            result.response.stream = wrapped.peekable().instrument(span);
        }

        Ok(result)
    }

    pub async fn start_batch_inference<'request>(
        &self,
        requests: &'request [ModelInferenceRequest<'request>],
        client: &'request TensorzeroHttpClient,
        api_keys: &'request InferenceCredentials,
    ) -> Result<StartBatchModelInferenceResponse, Error> {
        let mut provider_errors: IndexMap<String, Error> = IndexMap::new();
        for provider_name in &self.routing {
            let provider = self.providers.get(provider_name).ok_or_else(|| {
                Error::new(ErrorDetails::ProviderNotFound {
                    provider_name: provider_name.to_string(),
                })
            })?;
            let response = provider
                .start_batch_inference(requests, client, api_keys)
                .instrument(span!(
                    Level::INFO,
                    "start_batch_inference",
                    provider_name = &**provider_name
                ))
                .await;
            match response {
                Ok(response) => {
                    return Ok(StartBatchModelInferenceResponse::new(
                        response,
                        provider_name.clone(),
                        Arc::from(provider.provider_type()),
                    ));
                }
                Err(error) => {
                    provider_errors.insert(provider_name.to_string(), error);
                }
            }
        }
        Err(Error::new(ErrorDetails::AllModelProvidersFailed {
            provider_errors,
        }))
    }
}

/// Wraps a low-level model provider stream, adding in common functionality:
/// * Model inference cache writes
/// * OpenTelemetry usage attributes
///
/// This is used for functionality that needs access to individual chunks, which requires
/// us to wrap the underlying stream.
///
/// Note - this function is *not* called in relay mode
fn wrap_provider_stream(
    raw_request: String,
    model_request: ModelProviderRequest<'_>,
    ticket_borrow: TicketBorrows,
    clients: &InferenceClients,
    stream: Instrumented<PeekableProviderInferenceResponseStream>,
    write_to_cache: bool,
) -> Result<PeekableProviderInferenceResponseStream, Error> {
    // Detach the span from the stream, and re-attach it to the 'async_stream::stream!' wrapper
    // This ensures that the span duration include the entire provider-specific processing time
    let span = stream.span().clone();
    let mut stream = stream.into_inner();
    let cache_key = model_request.get_cache_key()?;
    let cache_manager = clients.cache_manager.clone();
    let tool_config = model_request
        .request
        .tool_config
        .clone()
        .map(std::borrow::Cow::into_owned);
    let otlp_config = clients.otlp_config.clone();
    let capture_content = otlp_config.genai_content_capture_enabled();
    let deferred_tasks = clients.deferred_tasks.clone();
    let span_clone = span.clone();
    let throughput_provider = model_request.provider_name.to_string();
    let throughput_model = model_request.model_name.to_string();
    let stream_started_at = tokio::time::Instant::now();

    // Collect usage objects for later use (e.g. rate limiting)
    let mut usages: Vec<Usage> = vec![];

    let base_stream = async_stream::stream! {
        let mut buffer = vec![];
        let mut errored = false;
        let should_buffer = write_to_cache || capture_content;

        // IMPORTANT: We should NOT modify chunks here, as they'll be re-processed downstream (e.g. `create_stream`).
        while let Some(chunk) = stream.next().await {
            if let Ok(chunk) = chunk.as_ref() && let Some(chunk_usage) = chunk.usage.as_ref() {
                usages.push(*chunk_usage);
            }

            // Buffer chunks when we need them downstream (cache write and/or GenAI span attribute emission).
            if should_buffer && !errored {
                match chunk.as_ref() {
                    Ok(chunk) => {
                        buffer.push(chunk.clone());
                    }
                    Err(e) => {
                        tracing::warn!("Skipping buffered chunk due to error in stream: {e}");
                        errored = true;
                    }
                }
            }

            // If we see a `FatalStreamError`, yield it and stop processing the stream,
            // to avoid holding open a stream that might never produce more chunks.
            // We'll still compute rate-limiting usage using all of the chunks that we've seen so far.
            if let Err(e) = chunk.as_ref()
                && let ErrorDetails::FatalStreamError { .. } = e.get_details()
            {
                errored = true;
                yield chunk;
                break;
            }

            yield chunk;
        }

        let aggregated_usage = aggregate_usage_from_single_streaming_model_inference(usages);

        if !errored {
            crate::routing::record_tokens_per_sec(
                &throughput_provider,
                &throughput_model,
                aggregated_usage.output_tokens,
                stream_started_at.elapsed(),
            );
        }

        otlp_config.apply_usage_to_model_provider_span(&span_clone, &aggregated_usage);
        if capture_content && !errored {
            let out_messages = genai_conventions::to_genai_output_from_chunks(&buffer);
            span_clone.set_attribute(
                "gen_ai.output.messages",
                serde_json::to_string(&out_messages).unwrap_or_default(),
            );
        }
        // Make sure that we finish updating rate-limiting tickets if the gateway shuts down
        deferred_tasks.spawn(async move {
            let nano_cost = aggregated_usage.cost.map(decimal_cost_to_nano_cost);
            let usage = match (aggregated_usage.total_tokens(), errored) {
                (Some(tokens), false) => {
                    RateLimitResourceUsage::Exact {
                        model_inferences: 1,
                        tokens: tokens as u64,
                        nano_cost,
                    }
                }
                _ => {
                    RateLimitResourceUsage::UnderEstimate {
                        model_inferences: 1,
                        tokens: aggregated_usage.total_tokens().unwrap_or(0) as u64,
                        nano_cost,
                    }
                }
            };

            record_usage_metrics(&aggregated_usage);

            if let Err(e) = ticket_borrow.return_tickets(usage).await {
                tracing::error!("Failed to return rate limit tickets: {}", e);
            }
        }.instrument(span_clone.clone()));

        if write_to_cache && !errored {
            let _ = start_cache_write_streaming(
                &cache_manager,
                cache_key,
                buffer,
                &raw_request,
                &aggregated_usage,
                tool_config
            );
        }
    }
    .instrument(span);
    // We unconditionally create a stream, and forward items into it from a separate task
    // This ensures that we keep processing chunks (and call `return_tickets` to update rate-limiting information)
    // even if the top-level HTTP request is later dropped.
    let (send, recv) = tokio::sync::mpsc::unbounded_channel();
    // Make sure that we finish processing the stream (so that we call `return_tickets` to update rate-limiting information)
    // if the gateway shuts down.
    clients.deferred_tasks.spawn(async move {
        futures::pin_mut!(base_stream);
        while let Some(chunk) = base_stream.next().await {
            // Intentionally ignore errors - the receiver might be dropped, but we want to keep polling
            // `base_stream` anyway (so that we compute the final usage and call `return_tickets`)
            let _ = send.send(chunk);
        }
    });
    Ok(
        (UnboundedReceiverStream::new(recv).boxed() as ProviderInferenceResponseStreamInner)
            .peekable(),
    )
}

#[derive(ts_rs::TS, Clone, Debug, Deserialize, PartialEq, Serialize)]
#[ts(export)]
pub struct UninitializedModelProvider {
    #[serde(flatten)]
    pub config: UninitializedProviderConfig,
    #[ts(skip)]
    pub extra_body: Option<ExtraBodyConfig>,
    #[ts(skip)]
    pub extra_headers: Option<ExtraHeadersConfig>,
    #[serde(default)]
    pub timeouts: TimeoutsConfig,
    /// If `true`, we emit a warning and discard chunks that we don't recognize
    /// (on a best-effort, per-provider basis).
    /// By default, unknown chunks are forwarded as-is in the stream.
    #[serde(default)]
    pub discard_unknown_chunks: bool,
    #[serde(default)]
    #[ts(skip)]
    pub cost: Option<UninitializedCostConfig>,
    #[serde(default)]
    #[ts(skip)]
    pub batch_cost: Option<UninitializedUnifiedCostConfig>,
    /// Default IANA timezone for `cost` peak windows that omit `timezone`.
    #[serde(default)]
    #[ts(skip)]
    pub timezone: Option<String>,
    /// ISO 4217 code for `cost` / `batch_cost` rates. Defaults to `USD`.
    #[serde(default)]
    #[ts(skip)]
    pub currency: Option<String>,
}

impl From<&UninitializedModelProvider> for StoredModelProvider {
    fn from(provider: &UninitializedModelProvider) -> Self {
        StoredModelProvider {
            provider: StoredProviderConfig::from(&provider.config),
            extra_body: provider
                .extra_body
                .as_ref()
                .map(StoredExtraBodyConfig::from),
            extra_headers: provider
                .extra_headers
                .as_ref()
                .map(StoredExtraHeadersConfig::from),
            timeouts: Some(StoredTimeoutsConfig::from(&provider.timeouts)),
            discard_unknown_chunks: Some(provider.discard_unknown_chunks),
            cost: provider.cost.as_ref().map(StoredCostConfig::from),
            batch_cost: provider
                .batch_cost
                .as_ref()
                .map(StoredUnifiedCostConfig::from),
            timezone: provider.timezone.clone(),
            currency: provider.currency.clone(),
        }
    }
}

#[derive(ts_rs::TS, Debug, Serialize)]
#[ts(export)]
pub struct ModelProvider {
    pub name: Arc<str>,
    pub config: ProviderConfig,
    #[ts(skip)]
    pub extra_headers: Option<ExtraHeadersConfig>,
    #[ts(skip)]
    pub extra_body: Option<ExtraBodyConfig>,
    pub timeouts: TimeoutsConfig,
    /// See `UninitializedModelProvider.discard_unknown_chunks`.
    pub discard_unknown_chunks: bool,
    #[serde(skip)]
    #[ts(skip)]
    pub cost: Option<CostConfig>,
    #[serde(skip)]
    #[ts(skip)]
    pub batch_cost: Option<CostConfig>,
}

impl ModelProvider {
    /// Returns the effective cost config for batch inferences.
    /// Uses `batch_cost` if configured, otherwise falls back to `cost`.
    pub fn effective_batch_cost_config(&self) -> Option<&CostConfig> {
        self.batch_cost.as_ref().or(self.cost.as_ref())
    }

    fn validate(&self, global_outbound_http_timeout: &chrono::Duration) -> Result<(), Error> {
        self.timeouts.validate(global_outbound_http_timeout)?;
        Ok(())
    }
    fn non_streaming_total_timeout(&self) -> Option<Duration> {
        Some(Duration::from_millis(
            self.timeouts.non_streaming.as_ref()?.total_ms?,
        ))
    }

    fn streaming_ttft_timeout(&self) -> Option<Duration> {
        Some(Duration::from_millis(
            self.timeouts.streaming.as_ref()?.ttft_ms?,
        ))
    }

    fn streaming_total_timeout(&self) -> Option<Duration> {
        Some(Duration::from_millis(
            self.timeouts.streaming.as_ref()?.total_ms?,
        ))
    }

    /// The name to report in the OTEL `gen_ai.system` attribute
    fn genai_system_name(&self) -> &'static str {
        self.provider_type()
    }

    /// The provider type string (e.g., "openai", "anthropic")
    pub fn provider_type(&self) -> &'static str {
        self.config.provider_type()
    }

    /// The API type used by this provider (e.g., ChatCompletions, Responses)
    pub fn api_type(&self) -> ApiType {
        self.config.api_type()
    }

    /// The model name to report in the OTEL `gen_ai.request.model` attribute
    fn genai_model_name(&self) -> Option<&str> {
        self.config.model_name()
    }
}

impl From<&ModelProvider> for ModelProviderRequestInfo {
    fn from(val: &ModelProvider) -> Self {
        ModelProviderRequestInfo {
            provider_name: val.name.clone(),
            extra_headers: val.extra_headers.clone(),
            extra_body: val.extra_body.clone(),
            discard_unknown_chunks: val.discard_unknown_chunks,
        }
    }
}

struct StreamAndRawRequest {
    stream: tracing_futures::Instrumented<PeekableProviderInferenceResponseStream>,
    raw_request: String,
    ticket_borrow: TicketBorrows,
}

pub struct StreamResponseAndMessages {
    pub response: StreamResponse,
    pub messages: Vec<RequestMessage>,
}

impl ModelProvider {
    fn apply_otlp_span_fields_input(&self, request: ModelProviderRequest<'_>, span: &Span) {
        let traces = match &request.otlp_config.traces {
            Some(t) => t,
            None => return,
        };
        if !traces.enabled.unwrap_or(false) {
            return;
        }
        match &traces.format {
            None | Some(OtlpTracesFormat::OpenTelemetry) => {
                span.set_attribute("gen_ai.operation.name", "chat");
                span.set_attribute("gen_ai.system", self.genai_system_name());

                if let Some(model_name) = self.genai_model_name() {
                    span.set_attribute("gen_ai.request.model", model_name.to_string());
                }

                if let Some(function_name) = request.function_name {
                    span.set_attribute("gen_ai.agent.name", function_name.to_string());
                }

                if traces.include_content.unwrap_or(false) {
                    let messages = genai_conventions::to_genai_messages(&request.request.messages);
                    span.set_attribute(
                        "gen_ai.input.messages",
                        serde_json::to_string(&messages).unwrap_or_default(),
                    );
                    if let Some(instructions) = genai_conventions::to_genai_system_instructions(
                        request.request.system.as_deref(),
                    ) {
                        span.set_attribute(
                            "gen_ai.system_instructions",
                            serde_json::to_string(&instructions).unwrap_or_default(),
                        );
                    }
                    if let Some(tool_defs) = genai_conventions::to_genai_tool_definitions(
                        request.request.tool_config.as_deref(),
                    ) {
                        span.set_attribute(
                            "gen_ai.tool.definitions",
                            serde_json::to_string(&tool_defs).unwrap_or_default(),
                        );
                    }
                }
            }
            Some(OtlpTracesFormat::OpenInference) => {
                span.set_attribute("openinference.span.kind", "LLM");
                span.set_attribute("llm.system", self.genai_system_name());

                if let Some(model_name) = self.genai_model_name() {
                    span.set_attribute("llm.model_name", model_name.to_string());
                }

                openinference_conventions::apply_input_messages(
                    span,
                    request.request.system.as_deref(),
                    &request.request.messages,
                );
            }
        }
    }

    #[expect(clippy::unused_self)] // We'll need 'self' for other attributes
    fn apply_otlp_span_fields_output(
        &self,
        otlp_config: &OtlpConfig,
        span: &Span,
        resp: &Result<ProviderInferenceResponse, Error>,
    ) {
        let traces_format = otlp_config.traces.as_ref().and_then(|t| t.format.clone());
        let include_content = otlp_config
            .traces
            .as_ref()
            .is_some_and(|t| t.enabled.unwrap_or(false) && t.include_content.unwrap_or(false));
        match resp {
            Ok(response) => {
                otlp_config.apply_usage_to_model_provider_span(span, &response.usage);
                match traces_format {
                    None | Some(OtlpTracesFormat::OpenTelemetry) => {
                        if include_content {
                            let out_messages = genai_conventions::to_genai_output(
                                &response.output,
                                response.finish_reason,
                            );
                            span.set_attribute(
                                "gen_ai.output.messages",
                                serde_json::to_string(&out_messages).unwrap_or_default(),
                            );
                        }
                    }
                    Some(OtlpTracesFormat::OpenInference) => {
                        // If we ever add providers that don't use JSON, we'll need to update this.
                        span.set_attribute("input.mime_type", "application/json");
                        span.set_attribute("input.value", response.raw_request.clone());
                        span.set_attribute("output.mime_type", "application/json");
                        span.set_attribute("output.value", response.raw_response.clone());
                    }
                }
            }
            Err(e) => {
                // If an error occurs, try to extract the raw request/response to attach to the OpenTelemetry span
                match e.get_details() {
                    ErrorDetails::InferenceClient {
                        raw_request,
                        raw_response,
                        ..
                    }
                    | ErrorDetails::InferenceServer {
                        raw_request,
                        raw_response,
                        ..
                    } => {
                        match traces_format {
                            None | Some(OtlpTracesFormat::OpenTelemetry) => {}
                            Some(OtlpTracesFormat::OpenInference) => {
                                // If we ever add providers that don't use JSON, we'll need to update this.
                                if let Some(raw_request) = raw_request {
                                    span.set_attribute("input.mime_type", "application/json");
                                    span.set_attribute("input.value", raw_request.clone());
                                }
                                if let Some(raw_response) = raw_response {
                                    span.set_attribute("output.mime_type", "application/json");
                                    span.set_attribute("output.value", raw_response.clone());
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    /// Validates that provider_tools are not used with providers that don't support them.
    /// Returns an error if dynamic provider_tools are scoped to this provider but it doesn't support them.
    fn validate_provider_tools_support(
        &self,
        request: &ProviderInferenceRequest<'_>,
    ) -> Result<(), Error> {
        // Skip validation if the provider supports provider_tools
        if self.config.supports_provider_tools() {
            return Ok(());
        }

        // Check if there are any dynamic provider_tools scoped to this provider
        if let Some(tool_config) = &request.request.tool_config {
            let scoped_provider_tools =
                tool_config.get_scoped_provider_tools(request.model_name, request.provider_name);
            if !scoped_provider_tools.is_empty() {
                return Err(Error::new(ErrorDetails::Config {
                    message: format!(
                        "Provider `{}` does not support `provider_tools`, but {} provider tool(s) were configured for model `{}` / provider `{}`. \
                        Provider tools are only supported by: Anthropic, GCP Vertex Anthropic, and OpenAI (Responses API only).",
                        self.config.thought_block_provider_type(),
                        scoped_provider_tools.len(),
                        request.model_name,
                        request.provider_name,
                    ),
                }));
            }
        }

        Ok(())
    }

    #[tracing::instrument(skip_all, fields(provider_name = &*self.name, otel.name = "model_provider_inference", stream = false))]
    async fn infer(
        &self,
        request: ModelProviderRequest<'_>,
        clients: &InferenceClients,
    ) -> Result<ProviderInferenceResponse, Error> {
        let span = Span::current();
        self.apply_otlp_span_fields_input(request, &span);

        let provider_request = ProviderInferenceRequest {
            request: request.request,
            model_name: request.model_name,
            provider_name: request.provider_name,
            model_inference_id: request.model_inference_id,
        };
        let model_provider_info = ModelProviderRequestInfo::from(self);

        // Validate that provider_tools are not used with unsupported providers
        self.validate_provider_tools_support(&provider_request)?;

        let ticket_borrow = clients
            .rate_limiting_manager
            .consume_tickets(&clients.scope_info, request.request)
            .await?;
        let res = self
            .config
            .infer(
                provider_request,
                &clients.http_client,
                &clients.credentials,
                &model_provider_info,
            )
            .await;
        self.apply_otlp_span_fields_output(request.otlp_config, &span, &res);
        let provider_inference_response = res?;
        record_usage_metrics(&provider_inference_response.usage);
        if let Ok(actual_resource_usage) = provider_inference_response.resource_usage() {
            // Make sure that we finish updating rate-limiting tickets if the gateway shuts down
            clients.deferred_tasks.spawn(
                async move {
                    if let Err(e) = ticket_borrow.return_tickets(actual_resource_usage).await {
                        tracing::error!("Failed to return rate limit tickets: {}", e);
                    }
                }
                .instrument(span),
            );
        }
        Ok(provider_inference_response)
    }

    #[tracing::instrument(skip_all, fields(provider_name = &*self.name, otel.name = "model_provider_inference", time_to_first_token, stream = true))]
    async fn infer_stream(
        &self,
        request: ModelProviderRequest<'_>,
        clients: &InferenceClients,
    ) -> Result<StreamAndRawRequest, Error> {
        self.apply_otlp_span_fields_input(request, &Span::current());

        let provider_request = ProviderInferenceRequest {
            request: request.request,
            model_name: request.model_name,
            provider_name: request.provider_name,
            model_inference_id: request.model_inference_id,
        };
        let model_provider_info = ModelProviderRequestInfo::from(self);

        // Validate that provider_tools are not used with unsupported providers
        self.validate_provider_tools_support(&provider_request)?;

        let ticket_borrow = clients
            .rate_limiting_manager
            .consume_tickets(&clients.scope_info, request.request)
            .await?;
        let (stream, raw_request) = self
            .config
            .infer_stream(
                provider_request,
                &clients.http_client,
                &clients.credentials,
                &model_provider_info,
            )
            .await?;

        // Attach the current `model_provider_inference` span to the stream.
        // This will cause the span to be entered every time the stream is polled,
        // extending the lifetime of the span in OpenTelemetry to include the entire
        // duration of the response stream.
        Ok(StreamAndRawRequest {
            stream: stream.instrument(Span::current()),
            raw_request,
            ticket_borrow,
        })
    }

    async fn start_batch_inference<'a>(
        &self,
        requests: &'a [ModelInferenceRequest<'a>],
        client: &'a TensorzeroHttpClient,
        api_keys: &'a InferenceCredentials,
    ) -> Result<StartBatchProviderInferenceResponse, Error> {
        self.config
            .start_batch_inference(requests, client, api_keys)
            .await
    }

    pub async fn poll_batch_inference<'a>(
        &self,
        batch_request: &'a BatchRequestRow<'_>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<PollBatchInferenceResponse, Error> {
        self.config
            .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
            .await
    }
}

pub use tensorzero_inference_types::credentials::{
    Credential, CredentialLocation, CredentialLocationOrHardcoded, CredentialLocationWithFallback,
    EndpointLocation, ModelProviderRequestInfo, ProviderInferenceRequest,
};

/// Default API roots for OpenAI-compatible Chinese providers.
/// TensorZero appends `/v1` (except Volcengine Ark, which already includes `/api/v3`).
pub(crate) const ALIBABA_DEFAULT_API_ROOT: &str = "https://dashscope.aliyuncs.com/compatible-mode";
pub(crate) const SILICONFLOW_DEFAULT_API_ROOT: &str = "https://api.siliconflow.cn";
pub(crate) const VOLCENGINE_DEFAULT_API_ROOT: &str = "https://ark.cn-beijing.volces.com/api/v3";

pub(crate) fn openai_compatible_shorthand_api_base_from_raw(
    raw: &str,
    append_openai_v1: bool,
) -> Result<Url, Error> {
    let mut raw = raw.trim().to_string();
    while raw.ends_with('/') {
        raw.pop();
    }
    if append_openai_v1 {
        let versioned = raw.ends_with("/v1")
            || raw.ends_with("/v3")
            || raw.contains("/v1/")
            || raw.contains("/v3/");
        if !versioned {
            raw.push_str("/v1");
        }
    }
    Url::parse(&raw).map_err(|e| {
        Error::new(ErrorDetails::Config {
            message: format!("Invalid OpenAI-compatible api_base `{raw}`: {e}"),
        })
    })
}

pub(crate) fn openai_compatible_shorthand_api_base(
    base_url_env: &str,
    default_root: &str,
    append_openai_v1: bool,
) -> Result<Url, Error> {
    let raw = std::env::var(base_url_env).unwrap_or_else(|_| default_root.to_string());
    openai_compatible_shorthand_api_base_from_raw(&raw, append_openai_v1)
}

pub(crate) async fn openai_compatible_shorthand_provider(
    model_name: String,
    base_url_env: &str,
    default_root: &str,
    append_openai_v1: bool,
    api_key_env: &str,
    default_credentials: &ProviderTypeDefaultCredentials,
) -> Result<OpenAIProvider, Error> {
    OpenAIProvider::new(
        model_name,
        Some(openai_compatible_shorthand_api_base(
            base_url_env,
            default_root,
            append_openai_v1,
        )?),
        OpenAIKind
            .get_defaulted_credential(
                Some(&CredentialLocationWithFallback::Single(
                    CredentialLocation::Env(api_key_env.to_string()),
                )),
                default_credentials,
            )
            .await?,
        OpenAIAPIType::ChatCompletions,
        false,
        Vec::new(),
        HashMap::new(),
    )
}

pub const SHORTHAND_MODEL_PREFIXES: &[&str] = &[
    "anthropic::",
    "deepseek::",
    "fireworks::",
    "google_ai_studio_gemini::",
    "gcp_vertex_gemini::",
    "gcp_vertex_anthropic::",
    "hyperbolic::",
    "groq::",
    "mistral::",
    "openai::",
    "openrouter::",
    "together::",
    "xai::",
    "alibaba::",
    "siliconflow::",
    "volcengine::",
    "dummy::",
];

pub type ModelTable = BaseModelTable<ModelConfig>;

impl ModelTable {
    /// Get the namespace of a statically-configured model, if any.
    /// Returns `None` if the model is not in the table or has no namespace.
    pub fn get_namespace(&self, model_name: &str) -> Option<&Namespace> {
        self.table
            .get(model_name)
            .and_then(|m| m.namespace.as_ref())
    }
}

impl ShorthandModelConfig for ModelConfig {
    const SHORTHAND_MODEL_PREFIXES: &[&str] = SHORTHAND_MODEL_PREFIXES;
    const MODEL_TYPE: &str = "Model";
    const TASK_TYPE: &str = "chat";
    async fn from_shorthand(
        provider_type: &str,
        model_name: &str,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self, Error> {
        let model_name = model_name.to_string();
        let provider_config = match provider_type {
            "anthropic" => ProviderConfig::Anthropic(AnthropicProvider::new(
                model_name,
                None,
                AnthropicKind
                    .get_defaulted_credential(None, default_credentials)
                    .await?,
                // No provider tools for shorthand models
                vec![],
            )),
            "deepseek" => ProviderConfig::DeepSeek(DeepSeekProvider::new(
                model_name,
                DeepSeekKind
                    .get_defaulted_credential(None, default_credentials)
                    .await?,
            )),
            "fireworks" => ProviderConfig::Fireworks(FireworksProvider::new(
                model_name,
                FireworksKind
                    .get_defaulted_credential(None, default_credentials)
                    .await?,
            )),
            "google_ai_studio_gemini" => {
                ProviderConfig::GoogleAIStudioGemini(GoogleAIStudioGeminiProvider::new(
                    model_name,
                    GoogleAIStudioGeminiKind
                        .get_defaulted_credential(None, default_credentials)
                        .await?,
                )?)
            }
            "gcp_vertex_gemini" => {
                let credentials = crate::model_table::GCPVertexGeminiKind
                    .get_defaulted_credential(None, default_credentials)
                    .await?;
                ProviderConfig::GCPVertexGemini(GCPVertexGeminiProvider::new_shorthand(
                    model_name,
                    credentials,
                )?)
            }
            "gcp_vertex_anthropic" => {
                let credentials = crate::model_table::GCPVertexAnthropicKind
                    .get_defaulted_credential(None, default_credentials)
                    .await?;
                ProviderConfig::GCPVertexAnthropic(GCPVertexAnthropicProvider::new_shorthand(
                    model_name,
                    credentials,
                )?)
            }
            "groq" => ProviderConfig::Groq(GroqProvider::new(
                model_name,
                GroqKind
                    .get_defaulted_credential(None, default_credentials)
                    .await?,
                None,
            )),
            "hyperbolic" => ProviderConfig::Hyperbolic(HyperbolicProvider::new(
                model_name,
                HyperbolicKind
                    .get_defaulted_credential(None, default_credentials)
                    .await?,
            )),
            "mistral" => ProviderConfig::Mistral(MistralProvider::new(
                model_name,
                MistralKind
                    .get_defaulted_credential(None, default_credentials)
                    .await?,
                None,
            )),
            "openai" => {
                if let Some(stripped_model_name) = model_name.strip_prefix("responses::") {
                    ProviderConfig::OpenAI(OpenAIProvider::new(
                        stripped_model_name.to_string(),
                        None,
                        OpenAIKind
                            .get_defaulted_credential(None, default_credentials)
                            .await?,
                        OpenAIAPIType::Responses,
                        false,
                        Vec::new(),
                        std::collections::HashMap::new(),
                    )?)
                } else {
                    ProviderConfig::OpenAI(OpenAIProvider::new(
                        model_name,
                        None,
                        OpenAIKind
                            .get_defaulted_credential(None, default_credentials)
                            .await?,
                        OpenAIAPIType::ChatCompletions,
                        false,
                        Vec::new(),
                        std::collections::HashMap::new(),
                    )?)
                }
            }
            "openrouter" => ProviderConfig::OpenRouter(OpenRouterProvider::new(
                model_name,
                OpenRouterKind
                    .get_defaulted_credential(None, default_credentials)
                    .await?,
            )),
            "together" => ProviderConfig::Together(TogetherProvider::new(
                model_name,
                TogetherKind
                    .get_defaulted_credential(None, default_credentials)
                    .await?,
            )),
            "xai" => ProviderConfig::XAI(XAIProvider::new(
                model_name,
                XAIKind
                    .get_defaulted_credential(None, default_credentials)
                    .await?,
            )),
            "alibaba" => ProviderConfig::OpenAI(
                openai_compatible_shorthand_provider(
                    model_name,
                    "ALIBABA_BASE_URL",
                    ALIBABA_DEFAULT_API_ROOT,
                    true,
                    "ALIBABA_API_KEY",
                    default_credentials,
                )
                .await?,
            ),
            "siliconflow" => ProviderConfig::OpenAI(
                openai_compatible_shorthand_provider(
                    model_name,
                    "SILICONFLOW_BASE_URL",
                    SILICONFLOW_DEFAULT_API_ROOT,
                    true,
                    "SILICONFLOW_API_KEY",
                    default_credentials,
                )
                .await?,
            ),
            "volcengine" => ProviderConfig::OpenAI(
                openai_compatible_shorthand_provider(
                    model_name,
                    "VOLCENGINE_BASE_URL",
                    VOLCENGINE_DEFAULT_API_ROOT,
                    false,
                    "VOLCENGINE_API_KEY",
                    default_credentials,
                )
                .await?
                .with_volcengine_audio_compat(),
            ),
            #[cfg(any(test, feature = "e2e_tests"))]
            "dummy" => ProviderConfig::Dummy(DummyProvider::new(model_name, None)?),
            _ => {
                return Err(ErrorDetails::Config {
                    message: format!("Invalid provider type: {provider_type}"),
                }
                .into());
            }
        };
        Ok(ModelConfig {
            routing: vec![provider_type.to_string().into()],
            providers: HashMap::from([(
                provider_type.to_string().into(),
                ModelProvider {
                    name: provider_type.into(),
                    config: provider_config,
                    extra_body: Default::default(),
                    extra_headers: Default::default(),
                    timeouts: Default::default(),
                    discard_unknown_chunks: false,
                    cost: None,
                    batch_cost: None,
                },
            )]),
            timeouts: Default::default(),
            skip_relay: false,
            namespace: None,
        })
    }

    fn merge_shorthand_targets(parts: Vec<(Arc<str>, Self)>) -> Result<Self, Error> {
        if parts.is_empty() {
            return Err(Error::new(ErrorDetails::Config {
                message: "Model alias has no shorthand targets".to_string(),
            }));
        }
        let mut routing = Vec::with_capacity(parts.len());
        let mut providers = HashMap::new();
        for (key, mut config) in parts {
            let inner_key = config.routing.first().cloned().ok_or_else(|| {
                Error::new(ErrorDetails::Config {
                    message: format!("Shorthand target `{key}` has empty routing"),
                })
            })?;
            let mut provider = config.providers.remove(&inner_key).ok_or_else(|| {
                Error::new(ErrorDetails::Config {
                    message: format!("Shorthand target `{key}` is missing its provider"),
                })
            })?;
            provider.name = key.clone();
            routing.push(key.clone());
            providers.insert(key, provider);
        }
        Ok(ModelConfig {
            routing,
            providers,
            timeouts: Default::default(),
            skip_relay: false,
            namespace: None,
        })
    }

    fn validate(
        &self,
        model_name: &str,
        global_outbound_http_timeout: &chrono::Duration,
    ) -> Result<(), Error> {
        self.timeouts.validate(global_outbound_http_timeout)?;
        // Ensure that the model has at least one provider
        if self.routing.is_empty() {
            return Err(ErrorDetails::Config {
                message: format!("`models.{model_name}`: `routing` must not be empty"),
            }
            .into());
        }

        // Ensure that routing entries are unique and exist as keys in providers
        let mut seen_providers = std::collections::HashSet::new();
        for provider in &self.routing {
            if provider.starts_with("tensorzero::") {
                return Err(ErrorDetails::Config {
                    message: format!("`models.{model_name}.routing`: Provider name cannot start with 'tensorzero::': {provider}"),
                }
                .into());
            }
            if !seen_providers.insert(provider) {
                return Err(ErrorDetails::Config {
                    message: format!("`models.{model_name}.routing`: duplicate entry `{provider}`"),
                }
                .into());
            }

            if !self.providers.contains_key(provider) {
                return Err(ErrorDetails::Config {
            message: format!(
                "`models.{model_name}`: `routing` contains entry `{provider}` that does not exist in `providers`"
            ),
        }
        .into());
            }
        }

        // Validate each provider
        for (provider_name, provider) in &self.providers {
            if !seen_providers.contains(provider_name) {
                return Err(ErrorDetails::Config {
                    message: format!(
                "`models.{model_name}`: Provider `{provider_name}` is not listed in `routing`"
            ),
                }
                .into());
            }
            provider.validate(global_outbound_http_timeout)?;
        }
        Ok(())
    }

    fn inherit_configured_provider_settings(
        &mut self,
        table: &HashMap<Arc<str>, Self>,
        requested_model_name: &str,
    ) {
        if table.is_empty() {
            return;
        }
        for provider in self.providers.values_mut() {
            if provider.cost.is_some() && provider.batch_cost.is_some() {
                continue;
            }
            let Some(source) =
                configured_provider_for_shorthand(table, requested_model_name, provider)
            else {
                continue;
            };
            if provider.cost.is_none() {
                provider.cost = source.cost.clone();
            }
            if provider.batch_cost.is_none() {
                provider.batch_cost = source.batch_cost.clone();
            }
        }
    }

    fn covers_shorthand_provider(&self, provider_type: &str) -> bool {
        self.routing
            .iter()
            .any(|name| crate::routing::routing_matches_requested(name, provider_type))
    }
}

fn configured_provider_for_shorthand<'a>(
    table: &'a HashMap<Arc<str>, ModelConfig>,
    requested_model_name: &str,
    shorthand: &ModelProvider,
) -> Option<&'a ModelProvider> {
    if let Some(configured) = table.get(requested_model_name)
        && let Some(provider) = configured.providers.values().find(|candidate| {
            candidate.name.as_ref() == shorthand.name.as_ref()
                || candidate.provider_type() == shorthand.provider_type()
        })
    {
        return Some(provider);
    }

    let billed = shorthand.genai_model_name()?;
    table.values().find_map(|configured| {
        configured.providers.values().find(|candidate| {
            candidate.provider_type() == shorthand.provider_type()
                && candidate.genai_model_name() == Some(billed)
        })
    })
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::sync::Arc;

    use crate::cache::{CacheEnabledMode, CacheManager};
    use crate::config::with_skip_credential_validation;
    use crate::model_alias::ModelAliasTable;
    use crate::rate_limiting::ScopeInfo;

    use crate::{
        cache::CacheOptions,
        db::{clickhouse::ClickHouseConnectionInfo, postgres::PostgresConnectionInfo},
        inference::types::{
            ContentBlockChunk, FunctionType, ModelInferenceRequestJsonMode, TextChunk,
        },
        model_table::RESERVED_MODEL_PREFIXES,
        providers::anthropic::AnthropicCredentials,
        providers::dummy::{
            DUMMY_INFER_RESPONSE_CONTENT, DUMMY_INFER_RESPONSE_RAW, DUMMY_STREAMING_RESPONSE,
            DummyCredentials,
        },
        rate_limiting::{RateLimitingConfig, RateLimitingManager, UninitializedRateLimitingConfig},
    };
    use secrecy::SecretString;
    use tensorzero_inference_types::ProviderToolCallConfig;
    use tokio_stream::StreamExt;
    use uuid::Uuid;

    use super::*;

    #[tokio::test]
    async fn test_model_config_infer_routing() {
        let good_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "good".into(),
            credentials: DummyCredentials::None,
        });
        let bad_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "error".into(),
            credentials: DummyCredentials::None,
        });
        let model_config = ModelConfig {
            routing: vec!["good_provider".into()],
            providers: HashMap::from([(
                "good_provider".into(),
                ModelProvider {
                    name: "good_provider".into(),
                    config: good_provider_config,
                    extra_body: Default::default(),
                    extra_headers: Default::default(),
                    timeouts: Default::default(),
                    discard_unknown_chunks: false,
                    cost: None,
                    batch_cost: None,
                },
            )]),
            timeouts: Default::default(),
            skip_relay: false,
            namespace: None,
        };
        let tool_config = ProviderToolCallConfig::default();
        let api_keys = InferenceCredentials::default();
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let clickhouse_connection_info = ClickHouseConnectionInfo::new_disabled();
        let clients = InferenceClients {
            failed_model_inference_datastore: None,
            http_client: http_client.clone(),
            clickhouse_connection_info: clickhouse_connection_info.clone(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            credentials: Arc::new(api_keys.clone()),
            cache_options: CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly,
            },
            cache_manager: CacheManager::new(Arc::new(clickhouse_connection_info.clone())),
            tags: Arc::new(Default::default()),
            rate_limiting_manager: Arc::new(RateLimitingManager::new_dummy()),
            otlp_config: Default::default(),
            deferred_tasks: tokio_util::task::TaskTracker::new(),
            scope_info: ScopeInfo {
                tags: Arc::new(HashMap::new()),
                api_key_public_id: None,
            },
            relay: None,
            include_raw_usage: false,
            include_raw_response: false,
            include_aggregated_response: false,
        };

        // Try inferring the good model only
        let request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![],
            system: None,
            tool_config: Some(Cow::Borrowed(&tool_config)),
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let model_name = "test model";
        let response = model_config
            .infer(&request, &clients, model_name, None)
            .await
            .unwrap();
        let content = response.output;
        assert_eq!(
            content,
            vec![DUMMY_INFER_RESPONSE_CONTENT.to_string().into()]
        );
        let raw = response.raw_response;
        assert_eq!(raw, DUMMY_INFER_RESPONSE_RAW);
        let usage = response.usage;
        assert_eq!(
            usage,
            Usage {
                input_tokens: Some(10),
                output_tokens: Some(1),
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: None,
                currency: None,
            }
        );
        assert_eq!(&*response.model_provider_name, "good_provider");

        // Try inferring the bad model
        let model_config = ModelConfig {
            routing: vec!["error".into()],
            providers: HashMap::from([(
                "error".into(),
                ModelProvider {
                    name: "error".into(),
                    config: bad_provider_config,
                    extra_body: Default::default(),
                    extra_headers: Default::default(),
                    timeouts: Default::default(),
                    discard_unknown_chunks: false,
                    cost: None,
                    batch_cost: None,
                },
            )]),
            timeouts: Default::default(),
            skip_relay: false,
            namespace: None,
        };
        let response = model_config
            .infer(&request, &clients, model_name, None)
            .await
            .unwrap_err();
        assert_eq!(
            response,
            ErrorDetails::AllModelProvidersFailed {
                provider_errors: IndexMap::from([(
                    "error".to_string(),
                    ErrorDetails::InferenceClient {
                        message: "Error sending request to Dummy provider for model 'error'."
                            .to_string(),
                        status_code: None,
                        provider_type: "dummy".to_string(),
                        api_type: ApiType::ChatCompletions,
                        raw_request: Some("raw request".to_string()),
                        raw_response: None,
                    }
                    .into()
                )])
            }
            .into()
        );
    }

    #[tokio::test]
    async fn test_model_provider_infer_max_tokens_check() {
        let provider = ModelProvider {
            name: "test_provider".into(),
            config: ProviderConfig::Dummy(DummyProvider {
                model_name: "good".into(),
                credentials: DummyCredentials::None,
            }),
            extra_body: Default::default(),
            extra_headers: Default::default(),
            timeouts: Default::default(),
            discard_unknown_chunks: false,
            cost: None,
            batch_cost: None,
        };

        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let clickhouse_connection_info = ClickHouseConnectionInfo::new_disabled();
        let postgres_mock = PostgresConnectionInfo::Disabled;
        let api_keys = InferenceCredentials::default();
        let tags = HashMap::new();

        // With token rate limiting enabled and no max_tokens
        let toml_str = r"
            [[rules]]
            tokens_per_second = 10
            always = true
        ";
        let toml_config: crate::config::rate_limiting::TomlUninitializedRateLimitingConfig =
            toml::from_str(toml_str).unwrap();
        let uninitialized_config: UninitializedRateLimitingConfig = toml_config.try_into().unwrap();
        let rate_limit_config: RateLimitingConfig = uninitialized_config.try_into().unwrap();

        let clients = InferenceClients {
            failed_model_inference_datastore: None,
            http_client: http_client.clone(),
            clickhouse_connection_info: clickhouse_connection_info.clone(),
            postgres_connection_info: postgres_mock.clone(),
            credentials: Arc::new(api_keys.clone()),
            cache_options: CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly,
            },
            cache_manager: CacheManager::new(Arc::new(clickhouse_connection_info.clone())),
            tags: Arc::new(tags.clone()),
            rate_limiting_manager: Arc::new(RateLimitingManager::new(
                Arc::new(rate_limit_config),
                Arc::new(postgres_mock.clone()),
            )),
            otlp_config: Default::default(),
            deferred_tasks: tokio_util::task::TaskTracker::new(),
            scope_info: ScopeInfo {
                tags: Arc::new(tags.clone()),
                api_key_public_id: None,
            },
            relay: None,
            include_raw_usage: false,
            include_raw_response: false,
            include_aggregated_response: false,
        };

        let request_no_max_tokens = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![],
            system: None,
            tool_config: None,
            temperature: None,
            max_tokens: None, // No max_tokens!
            ..Default::default()
        };

        let provider_request = ModelProviderRequest {
            request: &request_no_max_tokens,
            model_name: "test",
            provider_name: "test_provider",
            otlp_config: &Default::default(),
            model_inference_id: Uuid::now_v7(),
            function_name: None,
        };

        // Should fail with RateLimitMissingMaxTokens
        let result = provider.infer(provider_request, &clients).await;
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            Error::new(ErrorDetails::RateLimitMissingMaxTokens)
        );

        // With token rate limiting enabled and max_tokens provided
        let request_with_max_tokens = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![],
            system: None,
            tool_config: None,
            temperature: None,
            max_tokens: Some(100), // max_tokens provided
            ..Default::default()
        };

        // This should error because postgres is disabled, but it should not be the RateLimitMissingMaxTokens error
        let provider_request = ModelProviderRequest {
            request: &request_with_max_tokens,
            model_name: "test",
            provider_name: "test_provider",
            otlp_config: &Default::default(),
            model_inference_id: Uuid::now_v7(),
            function_name: None,
        };

        let result = provider
            .infer(provider_request, &clients)
            .await
            .unwrap_err();
        assert_ne!(result, Error::new(ErrorDetails::RateLimitMissingMaxTokens));
    }

    #[tokio::test]
    async fn test_model_config_infer_routing_fallback() {
        let logs_contain = crate::utils::testing::capture_logs();
        // Test that fallback works with bad --> good model provider

        let good_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "good".into(),
            credentials: DummyCredentials::None,
        });
        let bad_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "error".into(),
            credentials: DummyCredentials::None,
        });
        let api_keys = InferenceCredentials::default();
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let clickhouse_connection_info = ClickHouseConnectionInfo::new_disabled();
        let clients = InferenceClients {
            failed_model_inference_datastore: None,
            http_client: http_client.clone(),
            clickhouse_connection_info: clickhouse_connection_info.clone(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            credentials: Arc::new(api_keys.clone()),
            cache_options: CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly,
            },
            cache_manager: CacheManager::new(Arc::new(clickhouse_connection_info.clone())),
            tags: Arc::new(Default::default()),
            rate_limiting_manager: Arc::new(RateLimitingManager::new_dummy()),
            otlp_config: Default::default(),
            deferred_tasks: tokio_util::task::TaskTracker::new(),
            scope_info: ScopeInfo {
                tags: Arc::new(HashMap::new()),
                api_key_public_id: None,
            },
            relay: None,
            include_raw_usage: false,
            include_raw_response: false,
            include_aggregated_response: false,
        };
        // Try inferring the good model only
        let request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![],
            system: None,
            tool_config: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let model_config = ModelConfig {
            routing: vec![
                "error_provider".to_string().into(),
                "good_provider".to_string().into(),
            ],
            providers: HashMap::from([
                (
                    "error_provider".to_string().into(),
                    ModelProvider {
                        name: "error_provider".into(),
                        config: bad_provider_config,
                        extra_body: Default::default(),
                        extra_headers: Default::default(),
                        timeouts: Default::default(),
                        discard_unknown_chunks: false,
                        cost: None,
                        batch_cost: None,
                    },
                ),
                (
                    "good_provider".to_string().into(),
                    ModelProvider {
                        name: "good_provider".into(),
                        config: good_provider_config,
                        extra_body: Default::default(),
                        extra_headers: Default::default(),
                        timeouts: Default::default(),
                        discard_unknown_chunks: false,
                        cost: None,
                        batch_cost: None,
                    },
                ),
            ]),
            timeouts: Default::default(),
            skip_relay: false,
            namespace: None,
        };

        let model_name = "test model";
        let response = model_config
            .infer(&request, &clients, model_name, None)
            .await
            .unwrap();
        // Ensure that the error for the bad provider was logged, but the request worked nonetheless
        assert!(logs_contain(
            "Error sending request to Dummy provider for model 'error'."
        ));
        let content = response.output;
        assert_eq!(
            content,
            vec![DUMMY_INFER_RESPONSE_CONTENT.to_string().into()]
        );
        let raw = response.raw_response;
        assert_eq!(raw, DUMMY_INFER_RESPONSE_RAW);
        let usage = response.usage;
        assert_eq!(
            usage,
            Usage {
                input_tokens: Some(10),
                output_tokens: Some(1),
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: None,
                currency: None,
            }
        );
        assert_eq!(&*response.model_provider_name, "good_provider");
    }

    #[tokio::test]
    async fn test_alias_shorthand_targets_failover() {
        let creds = crate::model_table::ProviderTypeDefaultCredentials::default();
        let error = ModelConfig::from_shorthand("dummy", "error", &creds)
            .await
            .unwrap();
        let good = ModelConfig::from_shorthand("dummy", "good", &creds)
            .await
            .unwrap();
        let model = ModelConfig::merge_shorthand_targets(vec![
            ("dummy::error".into(), error),
            ("dummy::good".into(), good),
        ])
        .unwrap();
        assert_eq!(model.routing.len(), 2);

        let api_keys = InferenceCredentials::default();
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let clickhouse_connection_info = ClickHouseConnectionInfo::new_disabled();
        let clients = InferenceClients {
            failed_model_inference_datastore: None,
            http_client: http_client.clone(),
            clickhouse_connection_info: clickhouse_connection_info.clone(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            credentials: Arc::new(api_keys),
            cache_options: CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly,
            },
            cache_manager: CacheManager::new(Arc::new(clickhouse_connection_info.clone())),
            tags: Arc::new(Default::default()),
            rate_limiting_manager: Arc::new(RateLimitingManager::new_dummy()),
            otlp_config: Default::default(),
            deferred_tasks: tokio_util::task::TaskTracker::new(),
            scope_info: ScopeInfo {
                tags: Arc::new(HashMap::new()),
                api_key_public_id: None,
            },
            relay: None,
            include_raw_usage: false,
            include_raw_response: false,
            include_aggregated_response: false,
        };
        let request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![],
            system: None,
            tool_config: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let session = crate::routing::RoutingSession::new(false);
        let response = crate::routing::RoutingSession::scope(session.clone(), async {
            model
                .infer(&request, &clients, "alias_failover", None)
                .await
        })
        .await
        .unwrap();
        assert_eq!(&*response.model_provider_name, "dummy::good");
        let outcome = session.take_outcome().unwrap();
        assert_eq!(outcome.fallback_count, 1);
        assert_eq!(outcome.served_by, "dummy/good");

        let session = crate::routing::RoutingSession::new(true);
        let err = crate::routing::RoutingSession::scope(session, async {
            model
                .infer(&request, &clients, "alias_failover", None)
                .await
        })
        .await
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("Error sending request to Dummy provider for model 'error'")
        );
    }

    #[tokio::test]
    async fn test_model_config_infer_stream_routing() {
        let good_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "good".into(),
            credentials: DummyCredentials::None,
        });
        let bad_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "error".into(),
            credentials: DummyCredentials::None,
        });
        let api_keys = InferenceCredentials::default();
        let request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![],
            system: None,
            tool_config: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            seed: None,
            stream: true,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        // Test good model
        let model_config = ModelConfig {
            routing: vec!["good_provider".to_string().into()],
            providers: HashMap::from([(
                "good_provider".to_string().into(),
                ModelProvider {
                    name: "good_provider".into(),
                    config: good_provider_config,
                    extra_body: Default::default(),
                    extra_headers: Default::default(),
                    timeouts: Default::default(),
                    discard_unknown_chunks: false,
                    cost: None,
                    batch_cost: None,
                },
            )]),
            timeouts: Default::default(),
            skip_relay: false,
            namespace: None,
        };
        let clickhouse_connection_info = ClickHouseConnectionInfo::new_disabled();
        let clients = InferenceClients {
            failed_model_inference_datastore: None,
            http_client: TensorzeroHttpClient::new_testing().unwrap(),
            clickhouse_connection_info: clickhouse_connection_info.clone(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            credentials: Arc::new(api_keys.clone()),
            cache_options: CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::Off,
            },
            cache_manager: CacheManager::new(Arc::new(clickhouse_connection_info.clone())),
            tags: Arc::new(Default::default()),
            rate_limiting_manager: Arc::new(RateLimitingManager::new_dummy()),
            otlp_config: Default::default(),
            deferred_tasks: tokio_util::task::TaskTracker::new(),
            scope_info: ScopeInfo {
                tags: Arc::new(HashMap::new()),
                api_key_public_id: None,
            },
            relay: None,
            include_raw_usage: false,
            include_raw_response: false,
            include_aggregated_response: false,
        };
        let StreamResponseAndMessages {
            response:
                StreamResponse {
                    mut stream,
                    raw_request,
                    model_provider_name,
                    provider_type: _,
                    cached: _,
                    model_inference_id: _,
                    failed_raw_response: _,
                    cost_config: _,
                },
            messages: _input,
        } = model_config
            .infer_stream(&request, &clients, "my_model", None)
            .await
            .unwrap();
        let initial_chunk = stream.next().await.unwrap().unwrap();
        assert_eq!(
            initial_chunk.content,
            vec![ContentBlockChunk::Text(TextChunk {
                text: DUMMY_STREAMING_RESPONSE[0].to_string(),
                id: "0".to_string(),
            })],
        );
        assert_eq!(raw_request, "raw request");
        assert_eq!(&*model_provider_name, "good_provider");
        let mut collected_content: Vec<ContentBlockChunk> =
            vec![ContentBlockChunk::Text(TextChunk {
                text: DUMMY_STREAMING_RESPONSE[0].to_string(),
                id: "0".to_string(),
            })];
        let mut stream = Box::pin(stream);
        while let Some(Ok(chunk)) = stream.next().await {
            let mut content = chunk.content;
            assert!(content.len() <= 1);
            if content.len() == 1 {
                collected_content.push(content.pop().unwrap());
            }
        }
        let mut collected_content_str = String::new();
        for content in collected_content {
            match content {
                ContentBlockChunk::Text(text) => collected_content_str.push_str(&text.text),
                _ => panic!("Expected a text content block"),
            }
        }
        assert_eq!(collected_content_str, DUMMY_STREAMING_RESPONSE.join(""));

        // Test bad model
        let model_config = ModelConfig {
            routing: vec!["error".to_string().into()],
            providers: HashMap::from([(
                "error".to_string().into(),
                ModelProvider {
                    name: "error".to_string().into(),
                    config: bad_provider_config,
                    extra_body: Default::default(),
                    extra_headers: Default::default(),
                    timeouts: Default::default(),
                    discard_unknown_chunks: false,
                    cost: None,
                    batch_cost: None,
                },
            )]),
            timeouts: Default::default(),
            skip_relay: false,
            namespace: None,
        };
        let response = model_config
            .infer_stream(&request, &clients, "my_model", None)
            .await;
        assert!(response.is_err());
        let error = match response {
            Err(error) => error,
            Ok(_) => panic!("Expected error, got Ok(_)"),
        };
        assert_eq!(
            error,
            ErrorDetails::AllModelProvidersFailed {
                provider_errors: IndexMap::from([(
                    "error".to_string(),
                    ErrorDetails::InferenceClient {
                        message: "Error sending request to Dummy provider for model 'error'."
                            .to_string(),
                        status_code: None,
                        provider_type: "dummy".to_string(),
                        api_type: ApiType::ChatCompletions,
                        raw_request: Some("raw request".to_string()),
                        raw_response: None,
                    }
                    .into()
                )])
            }
            .into()
        );
    }

    #[tokio::test]
    async fn test_model_config_infer_stream_routing_fallback() {
        let logs_contain = crate::utils::testing::capture_logs();
        // Test that fallback works with bad --> good model provider (streaming)

        let good_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "good".into(),
            credentials: DummyCredentials::None,
        });
        let bad_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "error".into(),
            credentials: DummyCredentials::None,
        });
        let api_keys = InferenceCredentials::default();
        let request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![],
            system: None,
            tool_config: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            seed: None,
            stream: true,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        // Test fallback
        let model_config = ModelConfig {
            routing: vec!["error_provider".into(), "good_provider".into()],
            providers: HashMap::from([
                (
                    "error_provider".to_string().into(),
                    ModelProvider {
                        name: "error_provider".to_string().into(),
                        config: bad_provider_config,
                        extra_body: Default::default(),
                        extra_headers: Default::default(),
                        timeouts: Default::default(),
                        discard_unknown_chunks: false,
                        cost: None,
                        batch_cost: None,
                    },
                ),
                (
                    "good_provider".to_string().into(),
                    ModelProvider {
                        name: "good_provider".to_string().into(),
                        config: good_provider_config,
                        extra_body: Default::default(),
                        extra_headers: Default::default(),
                        timeouts: Default::default(),
                        discard_unknown_chunks: false,
                        cost: None,
                        batch_cost: None,
                    },
                ),
            ]),
            timeouts: Default::default(),
            skip_relay: false,
            namespace: None,
        };
        let clickhouse_connection_info = ClickHouseConnectionInfo::new_disabled();
        let clients = InferenceClients {
            failed_model_inference_datastore: None,
            http_client: TensorzeroHttpClient::new_testing().unwrap(),
            clickhouse_connection_info: clickhouse_connection_info.clone(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            credentials: Arc::new(api_keys.clone()),
            cache_options: CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::Off,
            },
            cache_manager: CacheManager::new(Arc::new(clickhouse_connection_info.clone())),
            tags: Arc::new(Default::default()),
            rate_limiting_manager: Arc::new(RateLimitingManager::new_dummy()),
            otlp_config: Default::default(),
            deferred_tasks: tokio_util::task::TaskTracker::new(),
            scope_info: ScopeInfo {
                tags: Arc::new(HashMap::new()),
                api_key_public_id: None,
            },
            relay: None,
            include_raw_usage: false,
            include_raw_response: false,
            include_aggregated_response: false,
        };
        let StreamResponseAndMessages {
            response:
                StreamResponse {
                    mut stream,
                    raw_request,
                    model_provider_name,
                    provider_type: _,
                    cached: _,
                    model_inference_id: _,
                    failed_raw_response: _,
                    cost_config: _,
                },
            messages: _,
        } = model_config
            .infer_stream(&request, &clients, "my_model", None)
            .await
            .unwrap();
        let initial_chunk = stream.next().await.unwrap().unwrap();
        assert_eq!(&*model_provider_name, "good_provider");
        // Ensure that the error for the bad provider was logged, but the request worked nonetheless
        assert!(logs_contain(
            "Error sending request to Dummy provider for model 'error'"
        ));
        assert_eq!(raw_request, "raw request");

        assert_eq!(
            initial_chunk.content,
            vec![ContentBlockChunk::Text(TextChunk {
                text: DUMMY_STREAMING_RESPONSE[0].to_string(),
                id: "0".to_string(),
            })],
        );

        let mut collected_content = initial_chunk.content;
        let mut stream = Box::pin(stream);
        while let Some(Ok(chunk)) = stream.next().await {
            let mut content = chunk.content;
            assert!(content.len() <= 1);
            if content.len() == 1 {
                collected_content.push(content.pop().unwrap());
            }
        }
        let mut collected_content_str = String::new();
        for content in collected_content {
            match content {
                ContentBlockChunk::Text(text) => collected_content_str.push_str(&text.text),
                _ => panic!("Expected a text content block"),
            }
        }
        assert_eq!(collected_content_str, DUMMY_STREAMING_RESPONSE.join(""));
    }

    #[tokio::test]
    async fn test_dynamic_api_keys() {
        let provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "test_key".into(),
            credentials: DummyCredentials::Dynamic("TEST_KEY".to_string()),
        });
        let model_config = ModelConfig {
            routing: vec!["model".into()],
            providers: HashMap::from([(
                "model".into(),
                ModelProvider {
                    name: "model".into(),
                    config: provider_config,
                    extra_body: Default::default(),
                    extra_headers: Default::default(),
                    timeouts: Default::default(),
                    discard_unknown_chunks: false,
                    cost: None,
                    batch_cost: None,
                },
            )]),
            timeouts: Default::default(),
            skip_relay: false,
            namespace: None,
        };
        let tool_config = ProviderToolCallConfig::default();
        let api_keys = InferenceCredentials::default();
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let clickhouse_connection_info = ClickHouseConnectionInfo::new_disabled();
        let clients = InferenceClients {
            failed_model_inference_datastore: None,
            http_client: http_client.clone(),
            clickhouse_connection_info: clickhouse_connection_info.clone(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            credentials: Arc::new(api_keys.clone()),
            cache_options: CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly,
            },
            cache_manager: CacheManager::new(Arc::new(clickhouse_connection_info.clone())),
            tags: Arc::new(Default::default()),
            rate_limiting_manager: Arc::new(RateLimitingManager::new_dummy()),
            otlp_config: Default::default(),
            deferred_tasks: tokio_util::task::TaskTracker::new(),
            scope_info: ScopeInfo {
                tags: Arc::new(HashMap::new()),
                api_key_public_id: None,
            },
            relay: None,
            include_raw_usage: false,
            include_raw_response: false,
            include_aggregated_response: false,
        };

        let request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![],
            system: None,
            tool_config: Some(Cow::Borrowed(&tool_config)),
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let model_name = "test model";
        let error = model_config
            .infer(&request, &clients, model_name, None)
            .await
            .unwrap_err();
        assert_eq!(
            error,
            ErrorDetails::AllModelProvidersFailed {
                provider_errors: IndexMap::from([(
                    "model".to_string(),
                    ErrorDetails::ApiKeyMissing {
                        provider_name: "Dummy".to_string(),
                        message: "Dynamic api key `TEST_KEY` is missing".to_string(),
                    }
                    .into()
                )])
            }
            .into()
        );

        let api_keys = HashMap::from([(
            "TEST_KEY".to_string(),
            SecretString::from("notgoodkey".to_string()),
        )]);
        let clients = InferenceClients {
            failed_model_inference_datastore: None,
            http_client: http_client.clone(),
            clickhouse_connection_info: clickhouse_connection_info.clone(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            credentials: Arc::new(api_keys.clone()),
            cache_options: CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly,
            },
            cache_manager: CacheManager::new(Arc::new(clickhouse_connection_info.clone())),
            tags: Arc::new(Default::default()),
            rate_limiting_manager: Arc::new(RateLimitingManager::new_dummy()),
            otlp_config: Default::default(),
            deferred_tasks: tokio_util::task::TaskTracker::new(),
            scope_info: ScopeInfo {
                tags: Arc::new(HashMap::new()),
                api_key_public_id: None,
            },
            relay: None,
            include_raw_usage: false,
            include_raw_response: false,
            include_aggregated_response: false,
        };
        let response = model_config
            .infer(&request, &clients, model_name, None)
            .await
            .unwrap_err();
        assert_eq!(
            response,
            ErrorDetails::AllModelProvidersFailed {
                provider_errors: IndexMap::from([(
                    "model".to_string(),
                    ErrorDetails::InferenceClient {
                        message: "Invalid API key for Dummy provider".to_string(),
                        status_code: None,
                        provider_type: "dummy".to_string(),
                        api_type: ApiType::ChatCompletions,
                        raw_request: Some("raw request".to_string()),
                        raw_response: None,
                    }
                    .into()
                )])
            }
            .into()
        );

        let provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "test_key".into(),
            credentials: DummyCredentials::Dynamic("TEST_KEY".to_string()),
        });
        let model_config = ModelConfig {
            routing: vec!["model".to_string().into()],
            providers: HashMap::from([(
                "model".to_string().into(),
                ModelProvider {
                    name: "model".to_string().into(),
                    config: provider_config,
                    extra_body: Default::default(),
                    extra_headers: Default::default(),
                    timeouts: Default::default(),
                    discard_unknown_chunks: false,
                    cost: None,
                    batch_cost: None,
                },
            )]),
            timeouts: Default::default(),
            skip_relay: false,
            namespace: None,
        };
        let tool_config = ProviderToolCallConfig::default();
        let api_keys = InferenceCredentials::default();
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let clickhouse_connection_info = ClickHouseConnectionInfo::new_disabled();
        let clients = InferenceClients {
            failed_model_inference_datastore: None,
            http_client: http_client.clone(),
            clickhouse_connection_info: clickhouse_connection_info.clone(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            credentials: Arc::new(api_keys.clone()),
            cache_options: CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly,
            },
            cache_manager: CacheManager::new(Arc::new(clickhouse_connection_info.clone())),
            tags: Arc::new(Default::default()),
            rate_limiting_manager: Arc::new(RateLimitingManager::new_dummy()),
            otlp_config: Default::default(),
            deferred_tasks: tokio_util::task::TaskTracker::new(),
            scope_info: ScopeInfo {
                tags: Arc::new(HashMap::new()),
                api_key_public_id: None,
            },
            relay: None,
            include_raw_usage: false,
            include_raw_response: false,
            include_aggregated_response: false,
        };

        let request = ModelInferenceRequest {
            messages: vec![],
            inference_id: Uuid::now_v7(),
            system: None,
            tool_config: Some(Cow::Borrowed(&tool_config)),
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let error = model_config
            .infer(&request, &clients, model_name, None)
            .await
            .unwrap_err();
        assert_eq!(
            error,
            ErrorDetails::AllModelProvidersFailed {
                provider_errors: IndexMap::from([(
                    "model".to_string(),
                    ErrorDetails::ApiKeyMissing {
                        provider_name: "Dummy".to_string(),
                        message: "Dynamic api key `TEST_KEY` is missing".to_string(),
                    }
                    .into()
                )])
            }
            .into()
        );

        let api_keys = HashMap::from([(
            "TEST_KEY".to_string(),
            SecretString::from("good_key".to_string()),
        )]);
        let clients = InferenceClients {
            failed_model_inference_datastore: None,
            http_client: http_client.clone(),
            clickhouse_connection_info: clickhouse_connection_info.clone(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            credentials: Arc::new(api_keys.clone()),
            cache_options: CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly,
            },
            cache_manager: CacheManager::new(Arc::new(clickhouse_connection_info.clone())),
            tags: Arc::new(Default::default()),
            rate_limiting_manager: Arc::new(RateLimitingManager::new_dummy()),
            otlp_config: Default::default(),
            deferred_tasks: tokio_util::task::TaskTracker::new(),
            scope_info: ScopeInfo {
                tags: Arc::new(HashMap::new()),
                api_key_public_id: None,
            },
            relay: None,
            include_raw_usage: false,
            include_raw_response: false,
            include_aggregated_response: false,
        };
        let response = model_config
            .infer(&request, &clients, model_name, None)
            .await
            .unwrap();
        assert_eq!(
            response.output,
            vec![DUMMY_INFER_RESPONSE_CONTENT.to_string().into()]
        );
    }

    #[tokio::test]
    async fn test_validate_or_create_model_config() {
        let model_table = ModelTable::default();
        // Test that we can get or create a model config
        model_table.validate("dummy::gpt-4o").unwrap();
        // Shorthand models are not added to the model table
        assert_eq!(model_table.static_model_len(), 0);
        let model_config = model_table
            .get("dummy::gpt-4o", None)
            .await
            .unwrap()
            .expect("Missing dummy model");
        assert_eq!(model_config.routing, vec![Arc::<str>::from("dummy")]);
        let provider_config = &model_config.providers.get("dummy").unwrap().config;
        match provider_config {
            ProviderConfig::Dummy(provider) => assert_eq!(&*provider.model_name, "gpt-4o"),
            _ => panic!("Expected Dummy provider"),
        }

        // Test that it fails if the model is not well-formed
        let model_config = model_table.validate("foo::bar");
        assert!(model_config.is_err());
        assert_eq!(
            model_config.unwrap_err(),
            ErrorDetails::Config {
                message: "Model name 'foo::bar' not found in model table".to_string()
            }
            .into()
        );
        // Test that it works with an initialized model
        let anthropic_provider_config = with_skip_credential_validation(async {
            ProviderConfig::Anthropic(AnthropicProvider::new(
                "claude".to_string(),
                None,
                AnthropicCredentials::None,
                vec![],
            ))
        })
        .await;
        let anthropic_model_config = ModelConfig {
            routing: vec!["anthropic".into()],
            providers: HashMap::from([(
                "anthropic".into(),
                ModelProvider {
                    name: "anthropic".into(),
                    config: anthropic_provider_config,
                    extra_body: Default::default(),
                    extra_headers: Default::default(),
                    timeouts: Default::default(),
                    discard_unknown_chunks: false,
                    cost: None,
                    batch_cost: None,
                },
            )]),
            timeouts: Default::default(),
            skip_relay: false,
            namespace: None,
        };
        let provider_types = ProviderTypesConfig::default();
        let model_table: ModelTable = ModelTable::new(
            HashMap::from([("claude".into(), anthropic_model_config)]),
            ProviderTypeDefaultCredentials::new(&provider_types).into(),
            chrono::Duration::seconds(120),
            Arc::new(ModelAliasTable::default()),
        )
        .unwrap();

        model_table.validate("dummy::claude").unwrap();
    }

    #[test]
    fn test_shorthand_prefixes_subset_of_reserved() {
        for &shorthand in SHORTHAND_MODEL_PREFIXES {
            assert!(
                RESERVED_MODEL_PREFIXES.contains(&shorthand.to_string()),
                "Shorthand prefix '{shorthand}' is not in RESERVED_MODEL_PREFIXES"
            );
        }
    }

    #[test]
    fn test_openai_compatible_api_base_appends_v1() {
        let alibaba =
            openai_compatible_shorthand_api_base_from_raw(ALIBABA_DEFAULT_API_ROOT, true).unwrap();
        assert_eq!(
            alibaba.as_str(),
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        );
        let already_versioned = openai_compatible_shorthand_api_base_from_raw(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/",
            true,
        )
        .unwrap();
        assert_eq!(
            already_versioned.as_str(),
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        );
        let volcengine =
            openai_compatible_shorthand_api_base_from_raw(VOLCENGINE_DEFAULT_API_ROOT, false)
                .unwrap();
        assert_eq!(
            volcengine.as_str(),
            "https://ark.cn-beijing.volces.com/api/v3"
        );
    }

    #[tokio::test]
    async fn test_china_provider_shorthand_validate_and_get() {
        let table = ModelTable::default();
        table.validate("alibaba::qwen-plus").unwrap();
        table.validate("siliconflow::Qwen/Qwen2-7B").unwrap();
        table.validate("volcengine::ep-xxx").unwrap();

        let model =
            with_skip_credential_validation(async { table.get("alibaba::qwen-plus", None).await })
                .await
                .unwrap()
                .unwrap();
        assert_eq!(model.routing.as_slice(), &[std::sync::Arc::from("alibaba")]);
        assert!(matches!(
            model.providers["alibaba"].config,
            crate::model::ProviderConfig::OpenAI(_)
        ));
    }

    #[test]
    fn test_credential_location_with_fallback_serialize_single() {
        // Test serializing a Single variant (backward compatible)
        let single =
            CredentialLocationWithFallback::Single(CredentialLocation::Env("API_KEY".to_string()));
        let json = serde_json::to_string(&single).unwrap();
        assert_eq!(json, r#""env::API_KEY""#);
    }

    #[test]
    fn test_credential_location_with_fallback_serialize_with_fallback() {
        // Test serializing a WithFallback variant
        let with_fallback = CredentialLocationWithFallback::WithFallback {
            default: CredentialLocation::Dynamic("key1".to_string()),
            fallback: CredentialLocation::Env("FALLBACK_KEY".to_string()),
        };
        let json = serde_json::to_string(&with_fallback).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["default"], "dynamic::key1");
        assert_eq!(parsed["fallback"], "env::FALLBACK_KEY");
    }

    #[test]
    fn test_credential_location_with_fallback_deserialize_single_string() {
        // Test deserializing from a simple string (backward compatible)
        let json = r#""env::API_KEY""#;
        let result: CredentialLocationWithFallback = serde_json::from_str(json).unwrap();
        match result {
            CredentialLocationWithFallback::Single(CredentialLocation::Env(key)) => {
                assert_eq!(key, "API_KEY");
            }
            _ => panic!("Expected Single(Env)"),
        }
    }

    #[test]
    fn test_credential_location_with_fallback_deserialize_dynamic_string() {
        // Test deserializing a dynamic credential from a string
        let json = r#""dynamic::my_key""#;
        let result: CredentialLocationWithFallback = serde_json::from_str(json).unwrap();
        match result {
            CredentialLocationWithFallback::Single(CredentialLocation::Dynamic(key)) => {
                assert_eq!(key, "my_key");
            }
            _ => panic!("Expected Single(Dynamic)"),
        }
    }

    #[test]
    fn test_credential_location_with_fallback_deserialize_with_fallback() {
        // Test deserializing an object with default and fallback fields
        let json = r#"{"default":"dynamic::key1","fallback":"env::FALLBACK_KEY"}"#;
        let result: CredentialLocationWithFallback = serde_json::from_str(json).unwrap();
        match result {
            CredentialLocationWithFallback::WithFallback { default, fallback } => {
                match default {
                    CredentialLocation::Dynamic(key) => assert_eq!(key, "key1"),
                    _ => panic!("Expected Dynamic for default"),
                }
                match fallback {
                    CredentialLocation::Env(key) => assert_eq!(key, "FALLBACK_KEY"),
                    _ => panic!("Expected Env for fallback"),
                }
            }
            CredentialLocationWithFallback::Single(..) => panic!("Expected WithFallback"),
        }
    }

    #[test]
    fn test_credential_location_with_fallback_deserialize_path_variants() {
        // Test deserializing path-based credentials
        let json = r#"{"default":"path::/etc/key","fallback":"path_from_env::KEY_PATH"}"#;
        let result: CredentialLocationWithFallback = serde_json::from_str(json).unwrap();
        match result {
            CredentialLocationWithFallback::WithFallback { default, fallback } => {
                match default {
                    CredentialLocation::Path(path) => assert_eq!(path, "/etc/key"),
                    _ => panic!("Expected Path for default"),
                }
                match fallback {
                    CredentialLocation::PathFromEnv(key) => assert_eq!(key, "KEY_PATH"),
                    _ => panic!("Expected PathFromEnv for fallback"),
                }
            }
            CredentialLocationWithFallback::Single(..) => panic!("Expected WithFallback"),
        }
    }

    #[test]
    fn test_credential_location_with_fallback_deserialize_sdk() {
        // Test deserializing SDK credential
        let json = r#""sdk""#;
        let result: CredentialLocationWithFallback = serde_json::from_str(json).unwrap();
        match result {
            CredentialLocationWithFallback::Single(CredentialLocation::Sdk) => {}
            _ => panic!("Expected Single(Sdk)"),
        }
    }

    #[test]
    fn test_credential_location_with_fallback_deserialize_none() {
        // Test deserializing None credential
        let json = r#""none""#;
        let result: CredentialLocationWithFallback = serde_json::from_str(json).unwrap();
        match result {
            CredentialLocationWithFallback::Single(CredentialLocation::None) => {}
            _ => panic!("Expected Single(None)"),
        }
    }

    #[test]
    fn test_credential_location_with_fallback_roundtrip_single() {
        // Test serializing and deserializing a Single variant
        let original =
            CredentialLocationWithFallback::Single(CredentialLocation::Env("MY_KEY".to_string()));
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: CredentialLocationWithFallback = serde_json::from_str(&json).unwrap();
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_credential_location_with_fallback_roundtrip_with_fallback() {
        // Test serializing and deserializing a WithFallback variant
        let original = CredentialLocationWithFallback::WithFallback {
            default: CredentialLocation::Dynamic("primary".to_string()),
            fallback: CredentialLocation::Env("SECONDARY".to_string()),
        };
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: CredentialLocationWithFallback = serde_json::from_str(&json).unwrap();
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_credential_location_with_fallback_deserialize_missing_default() {
        // Test that missing default field returns an error
        let json = r#"{"fallback":"env::FALLBACK_KEY"}"#;
        let result: Result<CredentialLocationWithFallback, _> = serde_json::from_str(json);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("missing field `default`")
        );
    }

    #[test]
    fn test_credential_location_with_fallback_deserialize_missing_fallback() {
        // Test that missing fallback field returns an error
        let json = r#"{"default":"env::DEFAULT_KEY"}"#;
        let result: Result<CredentialLocationWithFallback, _> = serde_json::from_str(json);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("missing field `fallback`")
        );
    }

    #[test]
    fn test_credential_location_with_fallback_deserialize_unknown_field() {
        // Test that unknown fields return an error
        let json = r#"{"default":"env::KEY","fallback":"env::FALLBACK","unknown":"value"}"#;
        let result: Result<CredentialLocationWithFallback, _> = serde_json::from_str(json);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("unknown field"));
    }

    #[test]
    fn test_credential_location_with_fallback_default_location() {
        // Test the default_location() method
        let single =
            CredentialLocationWithFallback::Single(CredentialLocation::Env("KEY".to_string()));
        match single.default_location() {
            CredentialLocation::Env(key) => assert_eq!(key, "KEY"),
            _ => panic!("Expected Env"),
        }

        let with_fallback = CredentialLocationWithFallback::WithFallback {
            default: CredentialLocation::Dynamic("primary".to_string()),
            fallback: CredentialLocation::Env("secondary".to_string()),
        };
        match with_fallback.default_location() {
            CredentialLocation::Dynamic(key) => assert_eq!(key, "primary"),
            _ => panic!("Expected Dynamic"),
        }
    }

    #[test]
    fn test_credential_location_with_fallback_fallback_location() {
        // Test the fallback_location() method
        let single =
            CredentialLocationWithFallback::Single(CredentialLocation::Env("KEY".to_string()));
        assert!(single.fallback_location().is_none());

        let with_fallback = CredentialLocationWithFallback::WithFallback {
            default: CredentialLocation::Dynamic("primary".to_string()),
            fallback: CredentialLocation::Env("secondary".to_string()),
        };
        match with_fallback.fallback_location() {
            Some(CredentialLocation::Env(key)) => assert_eq!(key, "secondary"),
            _ => panic!("Expected Some(Env)"),
        }
    }

    #[test]
    fn test_credential_location_rejects_hardcoded_values() {
        // CredentialLocation should reject hardcoded values (security requirement)
        // Only CredentialLocationOrHardcoded should accept them
        let json = r#""us-east-1""#;
        let result: Result<CredentialLocationWithFallback, _> = serde_json::from_str(json);
        assert!(
            result.is_err(),
            "CredentialLocation should reject hardcoded values like 'us-east-1'"
        );
        let err = result.unwrap_err();
        assert!(
            err.to_string()
                .contains("Invalid credential location format"),
            "Error message should mention invalid format, got: {err}"
        );
    }

    #[test]
    fn test_credential_location_or_hardcoded_accepts_hardcoded() {
        // CredentialLocationOrHardcoded should accept hardcoded values
        let json = r#""us-east-1""#;
        let result: CredentialLocationOrHardcoded = serde_json::from_str(json).unwrap();
        match result {
            CredentialLocationOrHardcoded::Hardcoded(value) => {
                assert_eq!(value, "us-east-1");
            }
            CredentialLocationOrHardcoded::Location(_) => panic!("Expected Hardcoded"),
        }
    }

    #[test]
    fn test_credential_location_or_hardcoded_accepts_env() {
        // CredentialLocationOrHardcoded should also accept env:: prefixed values
        let json = r#""env::AWS_REGION""#;
        let result: CredentialLocationOrHardcoded = serde_json::from_str(json).unwrap();
        #[expect(clippy::wildcard_enum_match_arm)]
        match result {
            CredentialLocationOrHardcoded::Location(CredentialLocation::Env(key)) => {
                assert_eq!(key, "AWS_REGION");
            }
            _ => panic!("Expected Location(Env)"),
        }
    }

    #[test]
    fn test_credential_location_or_hardcoded_accepts_dynamic() {
        // CredentialLocationOrHardcoded should accept dynamic:: prefixed values
        let json = r#""dynamic::region_key""#;
        let result: CredentialLocationOrHardcoded = serde_json::from_str(json).unwrap();
        #[expect(clippy::wildcard_enum_match_arm)]
        match result {
            CredentialLocationOrHardcoded::Location(CredentialLocation::Dynamic(key)) => {
                assert_eq!(key, "region_key");
            }
            _ => panic!("Expected Location(Dynamic)"),
        }
    }

    /// Test that `supports_provider_tools()` returns correct values for all providers.
    /// This test exists to ensure we don't forget to update the method when adding new providers.
    /// The exhaustive match in `supports_provider_tools()` provides compile-time safety,
    /// and this test verifies the expected runtime behavior.
    #[test]
    fn test_supports_provider_tools_coverage() {
        use crate::providers::dummy::DummyProvider;

        // Providers that SHOULD support provider_tools
        let anthropic = ProviderConfig::Anthropic(AnthropicProvider::new(
            "claude".to_string(),
            None,
            AnthropicCredentials::None,
            vec![],
        ));
        assert!(
            anthropic.supports_provider_tools(),
            "Anthropic should support provider_tools"
        );

        // Providers that should NOT support provider_tools
        let dummy = ProviderConfig::Dummy(DummyProvider::new("test".to_string(), None).unwrap());
        assert!(
            !dummy.supports_provider_tools(),
            "Dummy provider should not support provider_tools"
        );

        // Note: OpenAI support depends on api_type, tested separately in openai module
    }

    #[test]
    fn test_effective_batch_cost_config_returns_batch_cost_when_present() {
        use crate::cost::{CostConfig, CostConfigEntry, CostRate, NormalizedCostPointerConfig};
        use rust_decimal::Decimal;

        let cost_config: CostConfig = vec![CostConfigEntry {
            pointer: NormalizedCostPointerConfig::Unified {
                pointers: vec!["/usage/input_tokens".to_string()],
            },
            rate: Some(CostRate {
                cost_per_unit: Decimal::from(3) / Decimal::from(1_000_000),
            }),
            ..Default::default()
        }];

        let batch_cost_config: CostConfig = vec![CostConfigEntry {
            pointer: NormalizedCostPointerConfig::Unified {
                pointers: vec!["/usage/input_tokens".to_string()],
            },
            rate: Some(CostRate {
                cost_per_unit: Decimal::from(1) / Decimal::from(1_000_000),
            }),
            ..Default::default()
        }];

        let provider = ModelProvider {
            name: "test".into(),
            config: ProviderConfig::Dummy(DummyProvider {
                model_name: "test".into(),
                ..Default::default()
            }),
            extra_body: Default::default(),
            extra_headers: Default::default(),
            timeouts: Default::default(),
            discard_unknown_chunks: false,
            cost: Some(cost_config),
            batch_cost: Some(batch_cost_config),
        };

        let effective = provider
            .effective_batch_cost_config()
            .expect("should return batch_cost when present");
        assert_eq!(
            effective[0]
                .rate
                .as_ref()
                .expect("batch cost should have a flat rate")
                .cost_per_unit,
            Decimal::from(1) / Decimal::from(1_000_000),
            "should return batch_cost rate, not regular cost rate"
        );
    }

    #[test]
    fn test_effective_batch_cost_config_falls_back_to_cost() {
        use crate::cost::{CostConfig, CostConfigEntry, CostRate, NormalizedCostPointerConfig};
        use rust_decimal::Decimal;

        let cost_config: CostConfig = vec![CostConfigEntry {
            pointer: NormalizedCostPointerConfig::Unified {
                pointers: vec!["/usage/input_tokens".to_string()],
            },
            rate: Some(CostRate {
                cost_per_unit: Decimal::from(3) / Decimal::from(1_000_000),
            }),
            ..Default::default()
        }];

        let provider = ModelProvider {
            name: "test".into(),
            config: ProviderConfig::Dummy(DummyProvider {
                model_name: "test".into(),
                ..Default::default()
            }),
            extra_body: Default::default(),
            extra_headers: Default::default(),
            timeouts: Default::default(),
            discard_unknown_chunks: false,
            cost: Some(cost_config),
            batch_cost: None,
        };

        let effective = provider
            .effective_batch_cost_config()
            .expect("should fall back to cost when batch_cost is None");
        assert_eq!(
            effective[0]
                .rate
                .as_ref()
                .expect("cost should have a flat rate")
                .cost_per_unit,
            Decimal::from(3) / Decimal::from(1_000_000),
            "should return regular cost rate as fallback"
        );
    }

    #[test]
    fn test_effective_batch_cost_config_returns_none_when_both_absent() {
        let provider = ModelProvider {
            name: "test".into(),
            config: ProviderConfig::Dummy(DummyProvider {
                model_name: "test".into(),
                ..Default::default()
            }),
            extra_body: Default::default(),
            extra_headers: Default::default(),
            timeouts: Default::default(),
            discard_unknown_chunks: false,
            cost: None,
            batch_cost: None,
        };

        assert!(
            provider.effective_batch_cost_config().is_none(),
            "should return None when both cost and batch_cost are absent"
        );
    }

    // ─── Round-trip tests: Uninitialized → Stored → Uninitialized ───────────

    mod stored_round_trip {
        use std::collections::HashMap;
        use std::sync::Arc;

        use googletest::prelude::*;
        use rust_decimal::Decimal;
        use tensorzero_stored_config::{
            StoredCostConfig, StoredModelConfig, StoredModelProvider, StoredProviderConfig,
            StoredUnifiedCostConfig,
        };
        use tensorzero_types::{
            CostPointerConfig, PointerList, TierMode, UnifiedCostPointerConfig,
            UninitializedCostConfig, UninitializedCostConfigEntry, UninitializedCostRate,
            UninitializedCostTier, UninitializedPeakWindow, UninitializedPeakWindows,
            UninitializedUnifiedCostConfig,
        };

        use crate::config::{Namespace, NonStreamingTimeouts, TimeoutsConfig};
        use crate::inference::types::extra_body::{
            ExtraBodyConfig, ExtraBodyReplacement, ExtraBodyReplacementKind,
        };
        use crate::model::{
            CredentialLocation, CredentialLocationWithFallback, UninitializedModelConfig,
            UninitializedModelProvider, UninitializedProviderConfig,
        };
        use crate::providers::openai::OpenAIAPIType;

        #[gtest]
        fn test_credential_location_round_trip() {
            use tensorzero_stored_config::StoredCredentialLocation;
            let variants = vec![
                CredentialLocation::Env("MY_API_KEY".to_string()),
                CredentialLocation::Dynamic("x-custom-key".to_string()),
                CredentialLocation::Sdk,
                CredentialLocation::None,
            ];
            for original in &variants {
                let stored = StoredCredentialLocation::from(original);
                let restored = CredentialLocation::from(stored);
                expect_that!(restored, eq(original));
            }
        }

        #[gtest]
        fn test_credential_location_with_fallback_round_trip() {
            use tensorzero_stored_config::StoredCredentialLocationWithFallback;
            let variants = vec![
                CredentialLocationWithFallback::Single(CredentialLocation::Env(
                    "MY_KEY".to_string(),
                )),
                CredentialLocationWithFallback::WithFallback {
                    default: CredentialLocation::Dynamic("x-key".to_string()),
                    fallback: CredentialLocation::Env("FALLBACK_KEY".to_string()),
                },
            ];
            for original in &variants {
                let stored = StoredCredentialLocationWithFallback::from(original);
                let restored = CredentialLocationWithFallback::from(stored);
                expect_that!(restored, eq(original));
            }
        }

        #[gtest]
        fn test_provider_config_openai_round_trip() {
            let original = UninitializedProviderConfig::OpenAI {
                model_name: "gpt-4o".to_string(),
                api_base: None,
                api_key_location: Some(CredentialLocationWithFallback::Single(
                    CredentialLocation::Env("OPENAI_API_KEY".to_string()),
                )),
                api_type: OpenAIAPIType::ChatCompletions,
                include_encrypted_reasoning: false,
                provider_tools: vec![],
                content_type_overrides: HashMap::new(),
            };
            let stored = StoredProviderConfig::from(&original);
            let restored: UninitializedProviderConfig =
                stored.try_into().expect("should convert back");
            expect_that!(restored, eq(&original));
        }

        #[gtest]
        fn test_provider_config_anthropic_round_trip() {
            let original = UninitializedProviderConfig::Anthropic {
                model_name: "claude-sonnet-4-20250514".to_string(),
                api_base: None,
                api_key_location: Some(CredentialLocationWithFallback::Single(
                    CredentialLocation::Env("ANTHROPIC_API_KEY".to_string()),
                )),
                beta_structured_outputs: Some(true),
                provider_tools: vec![],
            };
            let stored = StoredProviderConfig::from(&original);
            let restored: UninitializedProviderConfig =
                stored.try_into().expect("should convert back");
            expect_that!(restored, eq(&original));
        }

        #[gtest]
        fn test_model_provider_round_trip() {
            let original = UninitializedModelProvider {
                config: UninitializedProviderConfig::OpenAI {
                    model_name: "gpt-4o".to_string(),
                    api_base: None,
                    api_key_location: Some(CredentialLocationWithFallback::Single(
                        CredentialLocation::Env("OPENAI_API_KEY".to_string()),
                    )),
                    api_type: OpenAIAPIType::ChatCompletions,
                    include_encrypted_reasoning: false,
                    provider_tools: vec![],
                    content_type_overrides: HashMap::new(),
                },
                extra_body: Some(ExtraBodyConfig {
                    data: vec![ExtraBodyReplacement {
                        pointer: "/temperature".to_string(),
                        kind: ExtraBodyReplacementKind::Value(serde_json::json!(0.5)),
                    }],
                }),
                extra_headers: None,
                timeouts: TimeoutsConfig {
                    non_streaming: Some(NonStreamingTimeouts {
                        total_ms: Some(30000),
                    }),
                    streaming: None,
                },
                discard_unknown_chunks: true,
                cost: Some(vec![UninitializedCostConfigEntry {
                    pointer: CostPointerConfig {
                        pointer: Some(PointerList::one("/usage/input_tokens")),
                        pointer_nonstreaming: None,
                        pointer_streaming: None,
                    },
                    rate: UninitializedCostRate {
                        cost_per_million: Some(Decimal::new(3, 0)),
                        cost_per_unit: None,
                    },
                    required: false,
                    usage: None,
                    peak: None,
                    ..Default::default()
                }]),
                batch_cost: None,
                timezone: None,
                currency: None,
            };
            let stored = StoredModelProvider::from(&original);
            let restored: UninitializedModelProvider =
                stored.try_into().expect("should convert back");
            expect_that!(restored, eq(&original));
        }

        #[gtest]
        fn test_model_config_round_trip() {
            let original = UninitializedModelConfig {
                routing: vec![Arc::from("provider_a")],
                providers: HashMap::from([(
                    Arc::from("provider_a"),
                    UninitializedModelProvider {
                        config: UninitializedProviderConfig::Anthropic {
                            model_name: "claude-sonnet-4-20250514".to_string(),
                            api_base: None,
                            api_key_location: Some(CredentialLocationWithFallback::Single(
                                CredentialLocation::Env("ANTHROPIC_API_KEY".to_string()),
                            )),
                            beta_structured_outputs: None,
                            provider_tools: vec![],
                        },
                        extra_body: None,
                        extra_headers: None,
                        timeouts: TimeoutsConfig::default(),
                        discard_unknown_chunks: false,
                        cost: None,
                        batch_cost: None,
                        timezone: None,
                        currency: None,
                    },
                )]),
                timeouts: TimeoutsConfig::default(),
                skip_relay: Some(true),
                namespace: Some(
                    Namespace::new("my_namespace".to_string()).expect("valid namespace"),
                ),
            };
            let stored = StoredModelConfig::try_from(&original).expect("should serialize");
            let restored: UninitializedModelConfig =
                stored.try_into().expect("should convert back");
            expect_that!(restored, eq(&original));
        }

        #[gtest]
        fn test_cost_config_round_trip() {
            let original: UninitializedCostConfig = vec![
                UninitializedCostConfigEntry {
                    pointer: CostPointerConfig {
                        pointer: Some(PointerList::one("/usage/input_tokens")),
                        pointer_nonstreaming: None,
                        pointer_streaming: Some(PointerList::one("/usage/streaming_tokens")),
                    },
                    rate: UninitializedCostRate {
                        cost_per_million: Some(Decimal::new(150, 2)),
                        cost_per_unit: None,
                    },
                    required: false,
                    usage: None,
                    peak: None,
                    ..Default::default()
                },
                UninitializedCostConfigEntry {
                    pointer: CostPointerConfig {
                        pointer: None,
                        pointer_nonstreaming: Some(PointerList::one("/usage/nonstream")),
                        pointer_streaming: None,
                    },
                    rate: UninitializedCostRate {
                        cost_per_million: None,
                        cost_per_unit: Some(Decimal::new(1, 6)),
                    },
                    required: true,
                    usage: None,
                    peak: None,
                    ..Default::default()
                },
            ];
            let stored = StoredCostConfig::from(&original);
            let restored: UninitializedCostConfig = stored.into();
            expect_that!(restored, eq(&original));
        }

        #[gtest]
        fn test_cost_config_tiers_and_multi_peak_round_trip() {
            let original: UninitializedCostConfig = vec![UninitializedCostConfigEntry {
                pointer: CostPointerConfig {
                    pointer: Some(PointerList::Many(vec![
                        "/usage/prompt_tokens".to_string(),
                        "/usage/input_tokens".to_string(),
                    ])),
                    pointer_nonstreaming: None,
                    pointer_streaming: None,
                },
                required: true,
                usage: Some(tensorzero_types::UsageField::Input),
                peak: Some(UninitializedPeakWindows::Many(vec![
                    UninitializedPeakWindow {
                        start: "09:00".to_string(),
                        end: "12:00".to_string(),
                        days: vec!["weekday".to_string()],
                        timezone: Some("Asia/Shanghai".to_string()),
                        rate: UninitializedCostRate {
                            cost_per_million: Some(Decimal::from(3)),
                            cost_per_unit: None,
                        },
                    },
                    UninitializedPeakWindow {
                        start: "14:00".to_string(),
                        end: "18:00".to_string(),
                        days: vec![],
                        timezone: None,
                        rate: UninitializedCostRate {
                            cost_per_million: Some(Decimal::from(3)),
                            cost_per_unit: None,
                        },
                    },
                ])),
                skip_if_pointer: Some(PointerList::one(
                    "/usage/completion_tokens_details/audio_tokens",
                )),
                tiers: vec![
                    UninitializedCostTier {
                        up_to: Some(32_000),
                        when: vec![],
                        rate: UninitializedCostRate {
                            cost_per_million: Some(Decimal::from(6)),
                            cost_per_unit: None,
                        },
                    },
                    UninitializedCostTier {
                        up_to: None,
                        when: vec![],
                        rate: UninitializedCostRate {
                            cost_per_million: Some(Decimal::from(8)),
                            cost_per_unit: None,
                        },
                    },
                ],
                tier_mode: TierMode::Progressive,
                tier_by: Some(PointerList::one("/usage/prompt_tokens")),
                ..Default::default()
            }];
            let stored = StoredCostConfig::from(&original);
            let restored: UninitializedCostConfig = stored.into();
            expect_that!(restored, eq(&original));
        }

        #[gtest]
        fn test_unified_cost_config_round_trip() {
            let original: UninitializedUnifiedCostConfig = vec![
                UninitializedCostConfigEntry {
                    pointer: UnifiedCostPointerConfig {
                        pointer: PointerList::one("/usage/input_tokens"),
                    },
                    rate: UninitializedCostRate {
                        cost_per_million: Some(Decimal::new(150, 2)),
                        cost_per_unit: None,
                    },
                    required: true,
                    usage: None,
                    peak: None,
                    ..Default::default()
                },
                UninitializedCostConfigEntry {
                    pointer: UnifiedCostPointerConfig {
                        pointer: PointerList::one("/usage/output_tokens"),
                    },
                    rate: UninitializedCostRate {
                        cost_per_million: None,
                        cost_per_unit: Some(Decimal::new(6, 6)),
                    },
                    required: false,
                    usage: None,
                    peak: None,
                    ..Default::default()
                },
            ];
            let stored = StoredUnifiedCostConfig::from(&original);
            let restored: UninitializedUnifiedCostConfig = stored.into();
            expect_that!(restored, eq(&original));
        }
    }
}

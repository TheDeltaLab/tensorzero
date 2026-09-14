// Modified by Delta-AI under Apache 2.0
#![recursion_limit = "256"]

use std::{collections::HashMap, sync::Arc};

use tensorzero_core::config::UninitializedConfig;
use tensorzero_core::config::snapshot::ConfigSnapshot;
use tensorzero_core::db::ConfigQueries;
use tensorzero_core::db::HealthCheckable;
use tensorzero_core::endpoints::stored_inferences::render_samples;
use tensorzero_core::error::{Error, ErrorDetails};
use tensorzero_core::stored_inference::StoredSample;
use uuid::Uuid;

// Re-export the core client from tensorzero-core

// Client core types
pub use tensorzero_core::client::{
    Client, ClientBuilder, ClientBuilderMode, ClientMode, EmbeddedGateway, HTTPGateway,
    PostgresConfig, get_config_no_verify_credentials,
};

// Client error types
pub use tensorzero_core::client::{
    ClientBuilderError, TensorZeroError, TensorZeroInternalError, err_to_http,
    with_embedded_timeout,
};

// Async inference client types (Delta-AI fork: `POST .../async` submit,
// `GET /v1/async_tasks/{task_id}` status and SSE stream)
pub use tensorzero_core::endpoints::status::StatusResponse;

// Client input types
pub use tensorzero_core::client::{
    CacheParamsOptions, ClientInferenceParams, ClientSecretString, Input, InputMessage,
    InputMessageContent,
};

// Input handling utilities
pub use tensorzero_core::client::input_handling;

// Re-export other commonly used types from tensorzero-core
pub use tensorzero_core::config::Config;
pub use tensorzero_core::db::clickhouse::query_builder::{
    BooleanMetricFilter, FloatComparisonOperator, FloatMetricFilter, InferenceFilter, OrderBy,
    OrderByTerm, OrderDirection, TagComparisonOperator, TagFilter, TimeComparisonOperator,
    TimeFilter,
};
pub use tensorzero_core::db::inferences::{InferenceOutputSource, ListInferencesParams};
pub use tensorzero_core::db::{
    ClickHouseConnection, EpisodeByIdRow, ModelUsageTimePoint, TableBoundsWithCount, TimeWindow,
};
pub use tensorzero_core::endpoints::episodes::internal::{
    ListEpisodesParams, ListEpisodesRequest, ListEpisodesResponse,
};
pub use tensorzero_core::endpoints::feedback::FeedbackResponse;
pub use tensorzero_core::endpoints::feedback::Params as FeedbackParams;
pub use tensorzero_core::endpoints::inference::{
    ChatCompletionInferenceParams, InferenceOutput, InferenceParams, InferenceResponse,
    InferenceResponseChunk, InferenceStream,
};
pub use tensorzero_core::endpoints::internal::config::{
    GetConfigResponse, WriteConfigRequest, WriteConfigResponse,
};
pub use tensorzero_core::endpoints::object_storage::ObjectResponse;
pub use tensorzero_core::endpoints::stored_inferences::v1::types::{
    GetInferencesRequest, GetInferencesResponse, ListInferencesRequest,
};
pub use tensorzero_core::inference::types::storage::{StorageKind, StoragePath};
pub use tensorzero_core::inference::types::{
    Base64File, ContentBlockChunk, File, ObjectStoragePointer, Role, System, Unknown, UnknownChunk,
    UrlFile, Usage,
};
pub use tensorzero_core::stored_inference::{
    RenderedSample, StoredChatInference, StoredChatInferenceDatabase, StoredInference,
    StoredInferenceDatabase, StoredJsonInference,
};
pub use tensorzero_core::tool::ToolCallWrapper;
pub use tensorzero_core::utils::gateway::setup_clickhouse_without_config;
pub use tensorzero_inference_types::tool::{DynamicToolParams, FunctionTool, Tool};

// Export quantile array from migration_0037
pub use tensorzero_core::db::clickhouse::migration_manager::migrations::migration_0037::QUANTILES;

// Re-export optimization types from tensorzero-optimizers
#[cfg(feature = "e2e_tests")]
pub mod test_helpers;

// Re-export observability for pyo3 feature
#[cfg(feature = "pyo3")]
pub use tensorzero_core::observability;




// NOTE(shuyangli): For methods that delegate to APIs in the gateway, the arguments generally are flattened from the request type for
// This is because when reading the code outside of an IDE, it's often difficult to tell the arguments apart without argument names.
//
// To illustrate:
//
// It's easy to understand the semantics of methods that take few, unambiguous arguments:
// ```rust
// ```
//
// But it quickly gets confusing with more arguments or arguments with similar types:
// ```rust
// ```
//
// In these cases, using the request type directly makes the code much more readable:
// ```rust
//     function_name: None,
//     limit: Some(100),
//     offset: Some(0),
//     filter: None,
// });
// ```

/// Extension trait for additional Client methods
#[async_trait::async_trait]
pub trait ClientExt {
    // ================================================================
    // Health checking
    // ================================================================
    async fn clickhouse_health(&self) -> Result<(), TensorZeroError>;

    // ================================================================
    // ================================================================












    // ================================================================
    // Inference operations
    // ================================================================
    #[cfg(feature = "e2e_tests")]
    async fn start_batch_inference(
        &self,
        params: tensorzero_core::endpoints::batch_inference::StartBatchInferenceParams,
    ) -> Result<
        tensorzero_core::endpoints::batch_inference::PrepareBatchInferenceOutput,
        TensorZeroError,
    >;

    /// Gets specific inferences by their IDs.
    ///
    /// # Arguments
    ///
    /// * `inference_ids` - The IDs of the inferences to retrieve.
    /// * `function_name` - Optional function name to filter by (improves query performance).
    /// * `output_source` - Whether to return inference or demonstration output.
    ///
    /// # Returns
    ///
    /// A `GetInferencesResponse` containing the requested inferences.
    ///
    /// # Errors
    ///
    /// Returns a `TensorZeroError` if the request fails.
    async fn get_inferences(
        &self,
        inference_ids: Vec<Uuid>,
        function_name: Option<String>,
        output_source: InferenceOutputSource,
    ) -> Result<GetInferencesResponse, TensorZeroError>;

    /// Lists inferences with optional filtering, pagination, and sorting.
    ///
    /// # Arguments
    ///
    /// * `request` - The request parameters for listing inferences.
    ///
    /// # Returns
    ///
    /// A `GetInferencesResponse` containing the inferences that match the criteria.
    ///
    /// # Errors
    ///
    /// Returns a `TensorZeroError` if the request fails.
    async fn list_inferences(
        &self,
        request: ListInferencesRequest,
    ) -> Result<GetInferencesResponse, TensorZeroError>;

    // ================================================================
    // Episode operations
    // ================================================================

    /// Lists episodes with pagination and optional filter support.
    ///
    /// # Arguments
    ///
    /// * `request` - The request parameters for listing episodes (limit, before, after, function_name, filters).
    ///
    /// # Returns
    ///
    /// A `ListEpisodesResponse` containing the episodes.
    ///
    /// # Errors
    ///
    /// Returns a `TensorZeroError` if the request fails.
    async fn list_episodes(
        &self,
        request: ListEpisodesRequest,
    ) -> Result<ListEpisodesResponse, TensorZeroError>;



    async fn experimental_render_samples<T: StoredSample + Send>(
        &self,
        stored_samples: Vec<T>,
        variants: HashMap<String, String>,
        concurrency: Option<usize>,
    ) -> Result<Vec<RenderedSample>, TensorZeroError>;





    // ================================================================
    // Config access
    // ================================================================
    fn config(&self) -> Option<Arc<Config>>;

    fn get_config(&self) -> Result<Arc<Config>, TensorZeroError>;

    /// Gets a config snapshot by hash, or the live config if no hash is provided.
    ///
    /// # Arguments
    ///
    /// * `hash` - Optional hash of the config snapshot to retrieve. If `None`, returns the live config.
    ///
    /// # Returns
    ///
    /// A `GetConfigResponse` containing the config snapshot.
    ///
    /// # Errors
    ///
    /// Returns a `TensorZeroError` if the request fails or the config snapshot is not found.
    async fn get_config_snapshot(
        &self,
        hash: Option<&str>,
    ) -> Result<GetConfigResponse, TensorZeroError>;

    /// Writes a config snapshot to the database.
    ///
    /// If a config with the same hash already exists, tags are merged
    /// (new tags override existing keys) and `created_at` is preserved.
    ///
    /// # Arguments
    ///
    /// * `request` - The config to write, including optional extra_templates and tags.
    ///
    /// # Returns
    ///
    /// A `WriteConfigResponse` containing the computed hash of the config.
    ///
    /// # Errors
    ///
    /// Returns a `TensorZeroError` if the request fails.
    async fn write_config(
        &self,
        request: WriteConfigRequest,
    ) -> Result<WriteConfigResponse, TensorZeroError>;


    #[cfg(any(feature = "e2e_tests", feature = "pyo3"))]
    #[expect(
        clippy::disallowed_types,
        reason = "e2e/pyo3 test helper that exposes the embedded gateway's SwappableAppStateData"
    )]
    fn get_app_state_data(&self)
    -> Option<&tensorzero_core::utils::gateway::SwappableAppStateData>;
}

#[async_trait::async_trait]
impl ClientExt for Client {
    /// Queries the health of the ClickHouse database
    /// This does nothing in `ClientMode::HTTPGateway`
    async fn clickhouse_health(&self) -> Result<(), TensorZeroError> {
        match self.mode() {
            ClientMode::HTTPGateway(_) => Ok(()),
            ClientMode::EmbeddedGateway {
                gateway,
                timeout: _,
            } => gateway
                .handle
                .app_state
                .clickhouse_connection_info()
                .health()
                .await
                .map_err(|e| TensorZeroError::Other {
                    source: e.log().into(),
                }),
        }
    }

    /// Gets the config from the embedded gateway
    /// Returns None for HTTP gateway mode
    fn config(&self) -> Option<Arc<Config>> {
        match self.mode() {
            ClientMode::HTTPGateway(_) => None,
            ClientMode::EmbeddedGateway { gateway, .. } => {
                Some(gateway.handle.app_state.config().load())
            }
        }
    }

    #[cfg(feature = "e2e_tests")]
    async fn start_batch_inference(
        &self,
        params: tensorzero_core::endpoints::batch_inference::StartBatchInferenceParams,
    ) -> Result<
        tensorzero_core::endpoints::batch_inference::PrepareBatchInferenceOutput,
        TensorZeroError,
    > {
        match self.mode() {
            ClientMode::HTTPGateway(_) => Err(TensorZeroError::Other {
                source: Error::new(ErrorDetails::InternalError {
                    message: "batch_inference is not yet implemented for HTTPGateway mode"
                        .to_string(),
                })
                .into(),
            }),
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                Ok(with_embedded_timeout(*timeout, async {
                    Box::pin(
                        tensorzero_core::endpoints::batch_inference::start_batch_inference(
                            gateway.handle.app_state.load_latest(),
                            params,
                            // We currently ban auth-enabled configs in embedded gateway mode,
                            // so we don't have an API key here
                            None,
                        ),
                    )
                    .await
                    .map_err(err_to_http)
                })
                .await?)
            }
        }
    }












    async fn get_inferences(
        &self,
        inference_ids: Vec<Uuid>,
        function_name: Option<String>,
        output_source: InferenceOutputSource,
    ) -> Result<GetInferencesResponse, TensorZeroError> {
        let request = GetInferencesRequest {
            ids: inference_ids,
            function_name,
            output_source,
        };
        match self.mode() {
            ClientMode::HTTPGateway(client) => {
                let url = client.base_url.join("v1/inferences/get_inferences").map_err(|e| TensorZeroError::Other {
                    source: Error::new(ErrorDetails::InvalidBaseUrl {
                        message: format!("Failed to join base URL with /v1/inferences/get_inferences endpoint: {e}"),
                    })
                    .into(),
                })?;
                let builder = client.http_client.post(url).json(&request);
                Ok(client.send_and_parse_http_response(builder).await?.0)
            }
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                with_embedded_timeout(*timeout, async {
                    let config = gateway.handle.app_state.config().load();
                    tensorzero_core::endpoints::stored_inferences::v1::get_inferences(
                        &config,
                        &gateway.handle.app_state.get_delegating_database(),
                        request,
                    )
                    .await
                    .map_err(err_to_http)
                })
                .await
            }
        }
    }

    async fn list_inferences(
        &self,
        request: ListInferencesRequest,
    ) -> Result<GetInferencesResponse, TensorZeroError> {
        match self.mode() {
            ClientMode::HTTPGateway(client) => {
                let url = client.base_url.join("v1/inferences/list_inferences").map_err(|e| TensorZeroError::Other {
                    source: Error::new(ErrorDetails::InvalidBaseUrl {
                        message: format!("Failed to join base URL with /v1/inferences/list_inferences endpoint: {e}"),
                    })
                    .into(),
                })?;
                let builder = client.http_client.post(url).json(&request);
                Ok(client.send_and_parse_http_response(builder).await?.0)
            }
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                with_embedded_timeout(*timeout, async {
                    let config = gateway.handle.app_state.config().load();
                    tensorzero_core::endpoints::stored_inferences::v1::list_inferences(
                        &config,
                        &gateway.handle.app_state.get_delegating_database(),
                        request,
                    )
                    .await
                    .map_err(err_to_http)
                })
                .await
            }
        }
    }

    async fn list_episodes(
        &self,
        request: ListEpisodesRequest,
    ) -> Result<ListEpisodesResponse, TensorZeroError> {
        match self.mode() {
            ClientMode::HTTPGateway(client) => {
                let url = client.base_url.join("internal/episodes").map_err(|e| {
                    TensorZeroError::Other {
                        source: Error::new(ErrorDetails::InvalidBaseUrl {
                            message: format!(
                                "Failed to join base URL with /internal/episodes endpoint: {e}"
                            ),
                        })
                        .into(),
                    }
                })?;
                let builder = client.http_client.post(url).json(&request);
                Ok(client.send_and_parse_http_response(builder).await?.0)
            }
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                let ListEpisodesRequest {
                    limit,
                    before,
                    after,
                    function_name,
                    filters,
                } = request;
                with_embedded_timeout(*timeout, async {
                    let config = gateway.handle.app_state.config().load();
                    let episodes = tensorzero_core::endpoints::episodes::internal::list_episodes(
                        &gateway.handle.app_state.get_delegating_database(),
                        &config,
                        limit,
                        before,
                        after,
                        function_name,
                        filters,
                    )
                    .await
                    .map_err(err_to_http)?;
                    Ok(ListEpisodesResponse { episodes })
                })
                .await
            }
        }
    }



    /// There are two things that need to happen in this function:
    /// 1. We need to resolve all network resources (e.g. images) in the inference examples.
    /// 2. We need to prepare all messages into "simple" messages that have been templated for a particular variant.
    ///    To do this, we need to know what variant to use for each function that might appear in the data.
    ///
    ///            has no variant specified, or where the process of downloading resources fails.
    ///            In future we will make this behavior configurable by the caller.
    async fn experimental_render_samples<T: StoredSample + Send>(
        &self,
        stored_samples: Vec<T>,
        variants: HashMap<String, String>, // Map from function name to variant name
        concurrency: Option<usize>,
    ) -> Result<Vec<RenderedSample>, TensorZeroError> {
        let ClientMode::EmbeddedGateway { gateway, .. } = self.mode() else {
            return Err(TensorZeroError::Other {
                source: Error::new(ErrorDetails::InvalidClientMode {
                    mode: "Http".to_string(),
                    message: "This function is only available in EmbeddedGateway mode".to_string(),
                })
                .into(),
            });
        };
        render_samples(
            gateway.handle.app_state.config().load(),
            stored_samples,
            variants,
            concurrency,
        )
        .await
        .map_err(err_to_http)
    }




    fn get_config(&self) -> Result<Arc<Config>, TensorZeroError> {
        match self.mode() {
            ClientMode::EmbeddedGateway { gateway, .. } => {
                Ok(gateway.handle.app_state.config().load())
            }
            ClientMode::HTTPGateway(_) => Err(TensorZeroError::Other {
                source: Error::new(ErrorDetails::InvalidClientMode {
                    mode: "Http".to_string(),
                    message: "This function is only available in EmbeddedGateway mode".to_string(),
                })
                .into(),
            }),
        }
    }

    async fn get_config_snapshot(
        &self,
        hash: Option<&str>,
    ) -> Result<GetConfigResponse, TensorZeroError> {
        match self.mode() {
            ClientMode::HTTPGateway(client) => {
                let endpoint = match hash {
                    Some(h) => format!("internal/config/{h}"),
                    None => "internal/config".to_string(),
                };
                let url = client
                    .base_url
                    .join(&endpoint)
                    .map_err(|e| TensorZeroError::Other {
                        source: Error::new(ErrorDetails::InvalidBaseUrl {
                            message: format!(
                                "Failed to join base URL with /{endpoint} endpoint: {e}"
                            ),
                        })
                        .into(),
                    })?;
                let builder = client.http_client.get(url);
                Ok(client.send_and_parse_http_response(builder).await?.0)
            }
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                with_embedded_timeout(*timeout, async {
                    let snapshot_hash = match hash {
                        Some(h) => h.parse().map_err(|_| {
                            err_to_http(Error::new(ErrorDetails::ConfigSnapshotNotFound {
                                snapshot_hash: h.to_string(),
                            }))
                        })?,
                        None => gateway.handle.app_state.config().load().hash.clone(),
                    };
                    let snapshot = gateway
                        .handle
                        .app_state
                        .get_delegating_database()
                        .get_config_snapshot(snapshot_hash)
                        .await
                        .map_err(err_to_http)?;
                    let uninitialized: UninitializedConfig =
                        snapshot.config.try_into().map_err(|e: &'static str| {
                            err_to_http(Error::new(ErrorDetails::Config {
                                message: e.to_string(),
                            }))
                        })?;
                    let config = serde_json::to_value(&uninitialized).map_err(|e| {
                        err_to_http(Error::new(ErrorDetails::Config {
                            message: format!("Failed to serialize config: {e}"),
                        }))
                    })?;
                    Ok(GetConfigResponse {
                        hash: snapshot.hash.to_string(),
                        config,
                        extra_templates: snapshot.extra_templates,
                        tags: snapshot.tags,
                    })
                })
                .await
            }
        }
    }

    async fn write_config(
        &self,
        request: WriteConfigRequest,
    ) -> Result<WriteConfigResponse, TensorZeroError> {
        match self.mode() {
            ClientMode::HTTPGateway(client) => {
                let url = client.base_url.join("internal/config").map_err(|e| {
                    TensorZeroError::Other {
                        source: Error::new(ErrorDetails::InvalidBaseUrl {
                            message: format!(
                                "Failed to join base URL with /internal/config endpoint: {e}"
                            ),
                        })
                        .into(),
                    }
                })?;
                let builder = client.http_client.post(url).json(&request);
                Ok(client.send_and_parse_http_response(builder).await?.0)
            }
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                Box::pin(with_embedded_timeout(*timeout, async {
                    let mut snapshot = ConfigSnapshot::new(request.config, request.extra_templates)
                        .map_err(err_to_http)?;
                    snapshot.tags = request.tags;

                    let hash = snapshot.hash.to_string();

                    gateway
                        .handle
                        .app_state
                        .validate_and_write_config_snapshot(&snapshot)
                        .await
                        .map_err(err_to_http)?;

                    Ok(WriteConfigResponse { hash })
                }))
                .await
            }
        }
    }



    #[cfg(any(feature = "e2e_tests", feature = "pyo3"))]
    #[expect(
        clippy::disallowed_types,
        reason = "e2e/pyo3 test helper that exposes the embedded gateway's SwappableAppStateData"
    )]
    fn get_app_state_data(
        &self,
    ) -> Option<&tensorzero_core::utils::gateway::SwappableAppStateData> {
        match self.mode() {
            ClientMode::EmbeddedGateway { gateway, .. } => Some(&gateway.handle.app_state),
            ClientMode::HTTPGateway(_) => None,
        }
    }
}




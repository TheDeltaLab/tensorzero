//! Delegating database connection that wraps both ClickHouse and Postgres.
//!
//! This module provides a database implementation that delegates operations
//! to either ClickHouse or Postgres based on the configured primary datastore.

use std::sync::LazyLock;

use async_trait::async_trait;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use uuid::Uuid;

use crate::config::ObservabilityBackend;
use crate::config::ObservabilityConfig;
use crate::config::snapshot::{ConfigSnapshot, SnapshotHash};
use crate::config::{Config, MetricConfigLevel};
use crate::db::BatchWriterHandle;
use crate::db::TimeWindow;
use crate::db::batch_inference::{BatchInferenceQueries, CompletedBatchInferenceRow};
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::db::clickhouse::clickhouse_client::ClickHouseClientType;
use crate::db::feedback::{
    BooleanMetricFeedbackInsert, CommentFeedbackInsert, CumulativeFeedbackTimeSeriesPoint,
    DemonstrationFeedbackInsert, DemonstrationFeedbackRow, FeedbackBounds, FeedbackByVariant,
    FeedbackQueries, FeedbackRow, FloatMetricFeedbackInsert, GetVariantPerformanceParams,
    LatestFeedbackRow, MetricWithFeedback, StaticEvaluationHumanFeedbackInsert,
    VariantPerformanceRow,
};
use crate::db::inferences::{
    CountByVariant, CountInferencesForFunctionParams, CountInferencesParams,
    CountInferencesWithFeedbackParams, FunctionInferenceCount, FunctionInfo,
    GetFunctionThroughputByVariantParams, InferenceMetadata, InferenceQueries,
    ListInferenceMetadataParams, ListInferencesParams, VariantThroughput,
};
use crate::db::model_inferences::ModelInferenceQueries;
use crate::db::postgres::PostgresConnectionInfo;
use crate::db::resolve_uuid::{ResolveUuidQueries, ResolvedObject};
use crate::db::variant_statistics::{
    GetVariantStatisticsParams, VariantStatisticsQueries, VariantStatisticsRow,
};
use crate::db::{
    CacheStatisticsTimePoint, ConfigQueries, DICLExampleWithDistance, DICLQueries,
    DeploymentIdQueries, EpisodeByIdRow, EpisodeQueries, HowdyFeedbackCounts, HowdyInferenceCounts,
    HowdyQueries, HowdyTokenUsage, ModelLatencyDatapoint, ModelUsageTimePoint, StoredDICLExample,
    TableBoundsWithCount, VariantUsageTimePoint,
};
use crate::endpoints::stored_inferences::v1::types::InferenceFilter;
use crate::error::{DelayedError, Error, ErrorDetails};
use crate::function::FunctionConfig;
use crate::inference::types::batch::{BatchModelInferenceRow, BatchRequestRow};
use crate::inference::types::{
    ChatInferenceDatabaseInsert, JsonInferenceDatabaseInsert, StoredModelInference,
};
use crate::stored_inference::StoredInferenceDatabase;
use crate::tool::ToolCallConfigDatabaseInsert;

/// Which database backend is the primary datastore for observability data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PrimaryDatastore {
    ClickHouse,
    Postgres,
    /// We should not write any observability data.
    Disabled,
}

impl PrimaryDatastore {
    /// Resolves the primary datastore from the observability config and available connections.
    ///
    /// Takes `observability.enabled` into account:
    /// - `Some(true)`: a backend **must** be available or this returns an error.
    /// - `None`: opportunistic — uses whatever is available, falls back to `Disabled`.
    /// - `Some(false)`: uses whatever is available for non-observability queries, falls back to `Disabled`.
    pub fn resolve(
        observability_config: &ObservabilityConfig,
        clickhouse: &ClickHouseConnectionInfo,
        postgres: &PostgresConnectionInfo,
    ) -> Result<Self, DelayedError> {
        let resolved = match observability_config.backend {
            None | Some(ObservabilityBackend::Auto) => {
                if clickhouse.client_type() != ClickHouseClientType::Disabled {
                    Self::ClickHouse
                } else if !matches!(postgres, PostgresConnectionInfo::Disabled) {
                    Self::Postgres
                } else {
                    Self::Disabled
                }
            }
            Some(ObservabilityBackend::ClickHouse) => Self::ClickHouse,
            Some(ObservabilityBackend::Postgres) => Self::Postgres,
        };

        match observability_config.enabled {
            Some(true) => match resolved {
                Self::Postgres => {
                    if matches!(postgres, PostgresConnectionInfo::Disabled) {
                        return Err(DelayedError::new(ErrorDetails::AppState {
                            message:
                                "A Postgres connection is required when the primary datastore \
                                 is Postgres and observability is enabled."
                                    .to_string(),
                        }));
                    }
                    Ok(Self::Postgres)
                }
                Self::ClickHouse => {
                    if clickhouse.client_type() == ClickHouseClientType::Disabled {
                        return Err(DelayedError::new(ErrorDetails::AppState {
                            message: "Missing environment variable `TENSORZERO_CLICKHOUSE_URL`."
                                .to_string(),
                        }));
                    }
                    Ok(Self::ClickHouse)
                }
                Self::Disabled => Err(DelayedError::new(ErrorDetails::AppState {
                    message: "Observability is enabled but no backend is available. \
                              Set `TENSORZERO_CLICKHOUSE_URL` or `TENSORZERO_POSTGRES_URL`, \
                              or configure `gateway.observability.backend` explicitly."
                        .to_string(),
                })),
            },
            None => {
                if resolved == Self::Disabled {
                    tracing::warn!(
                        "Disabling observability: `gateway.observability.enabled` is not explicitly enabled in the configuration and no backend is available (`TENSORZERO_CLICKHOUSE_URL` or `TENSORZERO_POSTGRES_URL`)."
                    );
                }
                Ok(resolved)
            }
            // TODO(#6469): audit uses of database operations when observability is disabled.
            Some(false) => Ok(Self::Disabled),
        }
    }

    /// Reads the primary datastore from `TENSORZERO_INTERNAL_TEST_OBSERVABILITY_BACKEND` env var.
    /// Returns `Postgres` if set to "postgres", otherwise `ClickHouse`. Never disabled.
    #[cfg(any(test, feature = "e2e_tests"))]
    pub fn from_test_env() -> Self {
        match std::env::var("TENSORZERO_INTERNAL_TEST_OBSERVABILITY_BACKEND").as_deref() {
            Ok("postgres") => PrimaryDatastore::Postgres,
            _ => PrimaryDatastore::ClickHouse,
        }
    }
}

/// A delegating database implementation that wraps both ClickHouse and Postgres.
///
/// Both ClickHouse and Postgres connections wrap an Arc<> under the hood, so this is safe and cheap to clone.
///
/// Routes all reads and writes to the configured `primary` datastore.
#[derive(Clone)]
pub struct DelegatingDatabaseConnection {
    pub clickhouse: ClickHouseConnectionInfo,
    pub postgres: PostgresConnectionInfo,
    primary: PrimaryDatastore,
}
/// A trait that allows us to express "The returned database supports all these queries"
/// via &(dyn DelegatingDatabaseQueries).
pub trait DelegatingDatabaseQueries:
    ConfigQueries
    + DeploymentIdQueries
    + HowdyQueries
    + FeedbackQueries
    + InferenceQueries
    + BatchInferenceQueries
    + ModelInferenceQueries
    + ResolveUuidQueries
    + EpisodeQueries
    + DICLQueries
    + VariantStatisticsQueries
{
    fn batcher_join_handles(&self) -> Vec<BatchWriterHandle>;
}
impl DelegatingDatabaseQueries for ClickHouseConnectionInfo {
    fn batcher_join_handles(&self) -> Vec<BatchWriterHandle> {
        self.batcher_join_handle().into_iter().collect()
    }
}
impl DelegatingDatabaseQueries for PostgresConnectionInfo {
    fn batcher_join_handles(&self) -> Vec<BatchWriterHandle> {
        self.batcher_join_handle().into_iter().collect()
    }
}

impl DelegatingDatabaseQueries for DelegatingDatabaseConnection {
    fn batcher_join_handles(&self) -> Vec<BatchWriterHandle> {
        let mut handles = Vec::new();
        match self.primary {
            PrimaryDatastore::Postgres => {
                if let Some(h) = self.postgres.batcher_join_handle() {
                    handles.push(h);
                }
            }
            PrimaryDatastore::ClickHouse => {
                if let Some(h) = self.clickhouse.batcher_join_handle() {
                    handles.push(h);
                }
            }
            PrimaryDatastore::Disabled => {}
        }
        handles
    }
}

impl DelegatingDatabaseConnection {
    pub fn new(
        clickhouse: ClickHouseConnectionInfo,
        postgres: PostgresConnectionInfo,
        primary: PrimaryDatastore,
    ) -> Self {
        Self {
            clickhouse,
            postgres,
            primary,
        }
    }

    fn get_database(&self) -> &(dyn DelegatingDatabaseQueries + Sync) {
        static DISABLED: LazyLock<ClickHouseConnectionInfo> =
            LazyLock::new(ClickHouseConnectionInfo::new_disabled);

        match self.primary {
            PrimaryDatastore::Postgres => &self.postgres,
            PrimaryDatastore::ClickHouse => &self.clickhouse,
            PrimaryDatastore::Disabled => &*DISABLED,
        }
    }
}

#[async_trait]
impl ConfigQueries for DelegatingDatabaseConnection {
    async fn get_config_snapshot(
        &self,
        snapshot_hash: SnapshotHash,
    ) -> Result<ConfigSnapshot, Error> {
        self.get_database().get_config_snapshot(snapshot_hash).await
    }

    async fn write_config_snapshot(&self, snapshot: &ConfigSnapshot) -> Result<(), DelayedError> {
        #[expect(clippy::disallowed_methods)]
        self.get_database().write_config_snapshot(snapshot).await
    }
}

#[async_trait]
impl DeploymentIdQueries for DelegatingDatabaseConnection {
    async fn get_deployment_id(&self) -> Result<String, DelayedError> {
        self.get_database().get_deployment_id().await
    }
}

#[async_trait]
impl HowdyQueries for DelegatingDatabaseConnection {
    async fn count_inferences_for_howdy(&self) -> Result<HowdyInferenceCounts, Error> {
        self.get_database().count_inferences_for_howdy().await
    }

    async fn count_feedbacks_for_howdy(&self) -> Result<HowdyFeedbackCounts, Error> {
        self.get_database().count_feedbacks_for_howdy().await
    }

    async fn get_token_totals_for_howdy(&self) -> Result<HowdyTokenUsage, Error> {
        self.get_database().get_token_totals_for_howdy().await
    }
}

#[async_trait]
impl FeedbackQueries for DelegatingDatabaseConnection {
    async fn get_feedback_by_variant(
        &self,
        metric_name: &str,
        function_name: &str,
        variant_names: Option<&Vec<String>>,
        namespace: Option<&str>,
        max_samples_per_variant: Option<u64>,
    ) -> Result<Vec<FeedbackByVariant>, Error> {
        self.get_database()
            .get_feedback_by_variant(
                metric_name,
                function_name,
                variant_names,
                namespace,
                max_samples_per_variant,
            )
            .await
    }

    async fn get_cumulative_feedback_timeseries(
        &self,
        function_name: String,
        metric_name: String,
        variant_names: Option<Vec<String>>,
        time_window: TimeWindow,
        max_periods: u32,
    ) -> Result<Vec<CumulativeFeedbackTimeSeriesPoint>, Error> {
        self.get_database()
            .get_cumulative_feedback_timeseries(
                function_name,
                metric_name,
                variant_names,
                time_window,
                max_periods,
            )
            .await
    }

    async fn query_feedback_by_target_id(
        &self,
        target_id: Uuid,
        before: Option<Uuid>,
        after: Option<Uuid>,
        limit: Option<u32>,
    ) -> Result<Vec<FeedbackRow>, Error> {
        self.get_database()
            .query_feedback_by_target_id(target_id, before, after, limit)
            .await
    }

    async fn query_feedback_bounds_by_target_id(
        &self,
        target_id: Uuid,
    ) -> Result<FeedbackBounds, Error> {
        self.get_database()
            .query_feedback_bounds_by_target_id(target_id)
            .await
    }

    async fn count_feedback_by_target_id(&self, target_id: Uuid) -> Result<u64, Error> {
        self.get_database()
            .count_feedback_by_target_id(target_id)
            .await
    }

    async fn query_demonstration_feedback_by_inference_id(
        &self,
        target_id: Uuid,
        before: Option<Uuid>,
        after: Option<Uuid>,
        limit: Option<u32>,
    ) -> Result<Vec<DemonstrationFeedbackRow>, Error> {
        self.get_database()
            .query_demonstration_feedback_by_inference_id(target_id, before, after, limit)
            .await
    }

    async fn query_metrics_with_feedback(
        &self,
        function_name: &str,
        function_config: &FunctionConfig,
        variant_name: Option<&str>,
    ) -> Result<Vec<MetricWithFeedback>, Error> {
        self.get_database()
            .query_metrics_with_feedback(function_name, function_config, variant_name)
            .await
    }

    async fn query_latest_feedback_id_by_metric(
        &self,
        target_id: Uuid,
    ) -> Result<Vec<LatestFeedbackRow>, Error> {
        self.get_database()
            .query_latest_feedback_id_by_metric(target_id)
            .await
    }

    async fn get_variant_performances(
        &self,
        params: GetVariantPerformanceParams<'_>,
    ) -> Result<Vec<VariantPerformanceRow>, Error> {
        self.get_database().get_variant_performances(params).await
    }

    async fn insert_boolean_feedback(
        &self,
        row: &BooleanMetricFeedbackInsert,
    ) -> Result<(), Error> {
        self.get_database().insert_boolean_feedback(row).await
    }

    async fn insert_float_feedback(&self, row: &FloatMetricFeedbackInsert) -> Result<(), Error> {
        self.get_database().insert_float_feedback(row).await
    }

    async fn insert_comment_feedback(&self, row: &CommentFeedbackInsert) -> Result<(), Error> {
        self.get_database().insert_comment_feedback(row).await
    }

    async fn insert_demonstration_feedback(
        &self,
        row: &DemonstrationFeedbackInsert,
    ) -> Result<(), Error> {
        self.get_database().insert_demonstration_feedback(row).await
    }

    async fn insert_static_eval_feedback(
        &self,
        row: &StaticEvaluationHumanFeedbackInsert,
    ) -> Result<(), Error> {
        self.get_database().insert_static_eval_feedback(row).await
    }
}

#[async_trait]
impl InferenceQueries for DelegatingDatabaseConnection {
    async fn list_inferences(
        &self,
        config: &Config,
        params: &ListInferencesParams<'_>,
    ) -> Result<Vec<StoredInferenceDatabase>, Error> {
        self.get_database().list_inferences(config, params).await
    }

    async fn list_inference_metadata(
        &self,
        params: &ListInferenceMetadataParams,
    ) -> Result<Vec<InferenceMetadata>, Error> {
        self.get_database().list_inference_metadata(params).await
    }

    async fn count_inferences(
        &self,
        config: &Config,
        params: &CountInferencesParams<'_>,
    ) -> Result<u64, Error> {
        self.get_database().count_inferences(config, params).await
    }

    async fn get_function_info(
        &self,
        target_id: &Uuid,
        level: MetricConfigLevel,
    ) -> Result<Option<FunctionInfo>, Error> {
        self.get_database()
            .get_function_info(target_id, level)
            .await
    }

    async fn get_chat_inference_tool_params(
        &self,
        function_name: &str,
        inference_id: Uuid,
    ) -> Result<Option<ToolCallConfigDatabaseInsert>, Error> {
        self.get_database()
            .get_chat_inference_tool_params(function_name, inference_id)
            .await
    }

    async fn get_json_inference_output_schema(
        &self,
        function_name: &str,
        inference_id: Uuid,
    ) -> Result<Option<Value>, Error> {
        self.get_database()
            .get_json_inference_output_schema(function_name, inference_id)
            .await
    }

    async fn get_serialized_inference_output_for_feedback(
        &self,
        function_info: &FunctionInfo,
        inference_id: Uuid,
    ) -> Result<Option<String>, Error> {
        self.get_database()
            .get_serialized_inference_output_for_feedback(function_info, inference_id)
            .await
    }

    async fn insert_chat_inferences(
        &self,
        rows: &[ChatInferenceDatabaseInsert],
    ) -> Result<(), Error> {
        if rows.is_empty() {
            return Ok(());
        }

        self.get_database().insert_chat_inferences(rows).await
    }

    async fn insert_json_inferences(
        &self,
        rows: &[JsonInferenceDatabaseInsert],
    ) -> Result<(), Error> {
        if rows.is_empty() {
            return Ok(());
        }

        self.get_database().insert_json_inferences(rows).await
    }

    // ===== Inference count methods (merged from InferenceCountQueries trait) =====

    async fn count_inferences_by_variant(
        &self,
        params: CountInferencesForFunctionParams<'_>,
    ) -> Result<Vec<CountByVariant>, Error> {
        self.get_database()
            .count_inferences_by_variant(params)
            .await
    }

    async fn count_inferences_with_feedback(
        &self,
        params: CountInferencesWithFeedbackParams<'_>,
    ) -> Result<u64, Error> {
        self.get_database()
            .count_inferences_with_feedback(params)
            .await
    }

    async fn get_function_throughput_by_variant(
        &self,
        params: GetFunctionThroughputByVariantParams<'_>,
    ) -> Result<Vec<VariantThroughput>, Error> {
        self.get_database()
            .get_function_throughput_by_variant(params)
            .await
    }

    async fn list_functions_with_inference_count(
        &self,
    ) -> Result<Vec<FunctionInferenceCount>, Error> {
        self.get_database()
            .list_functions_with_inference_count()
            .await
    }
}

#[async_trait]
impl BatchInferenceQueries for DelegatingDatabaseConnection {
    async fn get_batch_request(
        &self,
        batch_id: Uuid,
        inference_id: Option<Uuid>,
    ) -> Result<Option<BatchRequestRow<'static>>, Error> {
        self.get_database()
            .get_batch_request(batch_id, inference_id)
            .await
    }

    async fn get_batch_model_inferences(
        &self,
        batch_id: Uuid,
        inference_ids: &[Uuid],
    ) -> Result<Vec<BatchModelInferenceRow<'static>>, Error> {
        self.get_database()
            .get_batch_model_inferences(batch_id, inference_ids)
            .await
    }

    async fn get_completed_chat_batch_inferences(
        &self,
        batch_id: Uuid,
        function_name: &str,
        variant_name: &str,
        inference_id: Option<Uuid>,
    ) -> Result<Vec<CompletedBatchInferenceRow>, Error> {
        self.get_database()
            .get_completed_chat_batch_inferences(
                batch_id,
                function_name,
                variant_name,
                inference_id,
            )
            .await
    }

    async fn get_completed_json_batch_inferences(
        &self,
        batch_id: Uuid,
        function_name: &str,
        variant_name: &str,
        inference_id: Option<Uuid>,
    ) -> Result<Vec<CompletedBatchInferenceRow>, Error> {
        self.get_database()
            .get_completed_json_batch_inferences(
                batch_id,
                function_name,
                variant_name,
                inference_id,
            )
            .await
    }

    async fn write_batch_request(&self, row: &BatchRequestRow<'_>) -> Result<(), Error> {
        self.get_database().write_batch_request(row).await
    }

    async fn write_batch_model_inferences(
        &self,
        rows: &[BatchModelInferenceRow<'_>],
    ) -> Result<(), Error> {
        self.get_database().write_batch_model_inferences(rows).await
    }
}

#[async_trait]
impl ModelInferenceQueries for DelegatingDatabaseConnection {
    async fn get_model_inferences_by_inference_id(
        &self,
        inference_id: Uuid,
    ) -> Result<Vec<StoredModelInference>, Error> {
        self.get_database()
            .get_model_inferences_by_inference_id(inference_id)
            .await
    }

    async fn count_distinct_models_used(&self) -> Result<u32, Error> {
        self.get_database().count_distinct_models_used().await
    }

    async fn get_model_usage_timeseries(
        &self,
        time_window: TimeWindow,
        max_periods: u32,
    ) -> Result<Vec<ModelUsageTimePoint>, Error> {
        self.get_database()
            .get_model_usage_timeseries(time_window, max_periods)
            .await
    }

    async fn get_model_latency_quantiles(
        &self,
        time_window: TimeWindow,
    ) -> Result<Vec<ModelLatencyDatapoint>, Error> {
        self.get_database()
            .get_model_latency_quantiles(time_window)
            .await
    }

    fn get_model_latency_quantile_function_inputs(&self) -> &[f64] {
        self.get_database()
            .get_model_latency_quantile_function_inputs()
    }

    async fn get_variant_usage_timeseries(
        &self,
        function_name: &str,
        time_window: TimeWindow,
        max_periods: u32,
    ) -> Result<Vec<VariantUsageTimePoint>, Error> {
        self.get_database()
            .get_variant_usage_timeseries(function_name, time_window, max_periods)
            .await
    }

    async fn get_cache_statistics_timeseries(
        &self,
        time_window: TimeWindow,
        max_periods: u32,
        model_name: Option<&str>,
        model_provider_name: Option<&str>,
    ) -> Result<Vec<CacheStatisticsTimePoint>, Error> {
        self.get_database()
            .get_cache_statistics_timeseries(
                time_window,
                max_periods,
                model_name,
                model_provider_name,
            )
            .await
    }

    async fn insert_model_inferences(&self, rows: &[StoredModelInference]) -> Result<(), Error> {
        if rows.is_empty() {
            return Ok(());
        }

        self.get_database().insert_model_inferences(rows).await
    }
}


#[async_trait]
impl ResolveUuidQueries for DelegatingDatabaseConnection {
    async fn resolve_uuid(&self, id: &Uuid) -> Result<Vec<ResolvedObject>, Error> {
        self.get_database().resolve_uuid(id).await
    }
}

#[async_trait]
impl EpisodeQueries for DelegatingDatabaseConnection {
    async fn query_episode_table(
        &self,
        config: &Config,
        limit: u32,
        before: Option<Uuid>,
        after: Option<Uuid>,
        function_name: Option<String>,
        filters: Option<InferenceFilter>,
    ) -> Result<Vec<EpisodeByIdRow>, Error> {
        self.get_database()
            .query_episode_table(config, limit, before, after, function_name, filters)
            .await
    }

    async fn query_episode_table_bounds(&self) -> Result<TableBoundsWithCount, Error> {
        self.get_database().query_episode_table_bounds().await
    }
}

#[async_trait]
impl DICLQueries for DelegatingDatabaseConnection {
    async fn insert_dicl_example(&self, example: &StoredDICLExample) -> Result<(), Error> {
        self.get_database().insert_dicl_example(example).await
    }

    async fn insert_dicl_examples(&self, examples: &[StoredDICLExample]) -> Result<u64, Error> {
        self.get_database().insert_dicl_examples(examples).await
    }

    async fn get_similar_dicl_examples(
        &self,
        function_name: &str,
        variant_name: &str,
        embedding: &[f32],
        limit: u32,
    ) -> Result<Vec<DICLExampleWithDistance>, Error> {
        self.get_database()
            .get_similar_dicl_examples(function_name, variant_name, embedding, limit)
            .await
    }

    async fn has_dicl_examples(
        &self,
        function_name: &str,
        variant_name: &str,
    ) -> Result<bool, Error> {
        self.get_database()
            .has_dicl_examples(function_name, variant_name)
            .await
    }

    async fn delete_dicl_examples(
        &self,
        function_name: &str,
        variant_name: &str,
    ) -> Result<u64, Error> {
        self.get_database()
            .delete_dicl_examples(function_name, variant_name)
            .await
    }
}

#[async_trait]
impl VariantStatisticsQueries for DelegatingDatabaseConnection {
    async fn get_variant_statistics(
        &self,
        params: &GetVariantStatisticsParams,
    ) -> Result<Vec<VariantStatisticsRow>, Error> {
        self.get_database().get_variant_statistics(params).await
    }

    fn get_variant_statistics_quantiles(&self) -> Option<&[f64]> {
        self.get_database().get_variant_statistics_quantiles()
    }
}

#[cfg(any(test, feature = "e2e_tests"))]
mod test_helpers_impl {
    use super::{DelegatingDatabaseConnection, PrimaryDatastore};
    use crate::db::clickhouse::test_helpers::get_clickhouse;
    use crate::db::postgres::test_helpers::get_postgres;
    use crate::db::test_helpers::TestDatabaseHelpers;
    use async_trait::async_trait;

    impl DelegatingDatabaseConnection {
        pub async fn new_for_e2e_test() -> Self {
            let clickhouse = get_clickhouse().await;
            let postgres = get_postgres().await;
            let primary = PrimaryDatastore::from_test_env();
            Self::new(clickhouse, postgres, primary)
        }
    }

    #[async_trait]
    impl TestDatabaseHelpers for DelegatingDatabaseConnection {
        async fn flush_pending_writes(&self) {
            match self.primary {
                PrimaryDatastore::Postgres => self.postgres.flush_pending_writes().await,
                PrimaryDatastore::ClickHouse => self.clickhouse.flush_pending_writes().await,
                PrimaryDatastore::Disabled => {}
            }
        }

        async fn sleep_for_writes_to_be_visible(&self) {
            match self.primary {
                PrimaryDatastore::Postgres => {
                    self.postgres.sleep_for_writes_to_be_visible().await;
                }
                PrimaryDatastore::ClickHouse => {
                    self.clickhouse.sleep_for_writes_to_be_visible().await;
                }
                PrimaryDatastore::Disabled => {}
            }
        }

        async fn prepare_model_provider_statistics(&self) {
            match self.primary {
                PrimaryDatastore::Postgres => {
                    self.postgres.prepare_model_provider_statistics().await;
                }
                PrimaryDatastore::ClickHouse => {
                    self.clickhouse.prepare_model_provider_statistics().await;
                }
                PrimaryDatastore::Disabled => {}
            }
        }

        async fn prepare_variant_statistics(&self) {
            match self.primary {
                PrimaryDatastore::Postgres => {
                    self.postgres.prepare_variant_statistics().await;
                }
                PrimaryDatastore::ClickHouse => {
                    self.clickhouse.prepare_variant_statistics().await;
                }
                PrimaryDatastore::Disabled => {}
            }
        }
    }
}

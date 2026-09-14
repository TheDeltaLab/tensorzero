// Modified by Delta-AI under Apache 2.0
//! Internal route definitions for the TensorZero Gateway API.
//!
//! These routes are for internal use. They are unstable and might change without notice,
//! and do not export any OpenTelemetry spans.

use axum::{
    Router,
    routing::{get, post},
};
use tensorzero_core::endpoints;
use tensorzero_core::feature_flags;
#[expect(
    clippy::disallowed_types,
    reason = "router builders are parameterized on SwappableAppStateData by axum's type system"
)]
use tensorzero_core::utils::gateway::SwappableAppStateData;

#[expect(
    clippy::disallowed_types,
    reason = "router builders are parameterized on SwappableAppStateData by axum's type system"
)]
pub fn build_internal_non_otel_enabled_routes() -> Router<SwappableAppStateData> {
    let router = Router::new()
        .route(
            "/internal/functions/{function_name}/metrics",
            get(endpoints::functions::internal::get_function_metrics_handler),
        )
        .route(
            "/internal/functions/{function_name}/variant_performances",
            get(endpoints::functions::internal::get_variant_performances_handler),
        )
        .route(
            "/internal/functions/inference_counts",
            get(endpoints::internal::inference_count::list_functions_with_inference_count_handler),
        )
        .route(
            "/internal/inference_api_keys",
            get(endpoints::internal::inference_api_keys::list_inference_api_keys_handler),
        )
        .route(
            "/internal/dashboard/session",
            get(endpoints::internal::dashboard::get_dashboard_session_handler),
        )
        .route(
            "/internal/dashboard/users",
            get(endpoints::internal::dashboard::list_dashboard_users_handler)
                .post(endpoints::internal::dashboard::create_dashboard_user_handler)
                .patch(endpoints::internal::dashboard::update_dashboard_user_handler),
        )
        .route(
            "/internal/dashboard/users/delete",
            post(endpoints::internal::dashboard::delete_dashboard_user_handler),
        )
        .route(
            "/internal/functions/{function_name}/inference_count",
            get(endpoints::internal::inference_count::get_inference_count_handler),
        )
        .route(
            "/internal/functions/{function_name}/inference_count/{metric_name}",
            get(endpoints::internal::inference_count::get_inference_with_feedback_count_handler),
        )
        .route(
            "/internal/feedback/{target_id}",
            get(endpoints::feedback::internal::get_feedback_by_target_id_handler),
        )
        .route(
            "/internal/feedback/{target_id}/bounds",
            get(endpoints::feedback::internal::get_feedback_bounds_by_target_id_handler),
        )
        .route(
            "/internal/feedback/{target_id}/latest_id_by_metric",
            get(endpoints::feedback::internal::get_latest_feedback_id_by_metric_handler),
        )
        .route(
            "/internal/feedback/{target_id}/count",
            get(endpoints::feedback::internal::count_feedback_by_target_id_handler),
        )
        .route(
            "/internal/feedback/timeseries",
            get(endpoints::feedback::internal::get_cumulative_feedback_timeseries_handler),
        )
        .route(
            "/internal/feedback/{inference_id}/demonstrations",
            get(endpoints::feedback::internal::get_demonstration_feedback_handler),
        )
        .route(
            "/internal/functions/{function_name}/throughput_by_variant",
            get(endpoints::internal::inference_count::get_function_throughput_by_variant_handler),
        )
        .route(
            "/internal/functions/{function_name}/variant_usage",
            get(endpoints::internal::models::get_variant_usage_handler),
        )
        .route(
            "/internal/model_inferences/{inference_id}",
            get(endpoints::internal::model_inferences::get_model_inferences_handler),
        )
        .route(
            "/internal/inference_metadata",
            get(endpoints::internal::inference_metadata::get_inference_metadata_handler),
        )
        // Inference storage stats and retention endpoints
        .route(
            "/internal/inference_storage/stats",
            get(endpoints::internal::inference_storage::get_inference_storage_stats_handler),
        )
        .route(
            "/internal/inference_storage/retention",
            post(endpoints::internal::inference_storage::update_inference_retention_handler),
        )
        // Inference protection endpoints
        .route(
            "/internal/inferences/{inference_id}/protection",
            post(endpoints::internal::inference_protection::set_inference_protection_handler),
        )
        .route(
            "/internal/inferences/protection",
            post(endpoints::internal::inference_protection::get_inferences_protection_handler),
        )
        .route(
            "/internal/ui_config",
            get(endpoints::ui::get_config::ui_config_handler),
        )
        .route(
            "/internal/ui_config/{hash}",
            get(endpoints::ui::get_config::ui_config_by_hash_handler),
        )
        .route(
            "/internal/episodes",
            get(endpoints::episodes::internal::list_episodes_handler)
                .post(endpoints::episodes::internal::list_episodes_post_handler),
        )
        .route(
            "/internal/episodes/bounds",
            get(endpoints::episodes::internal::query_episode_table_bounds_handler),
        )
        .route(
            "/internal/episodes/{episode_id}/inference_count",
            get(endpoints::episodes::internal::get_episode_inference_count_handler),
        )
        .route(
            "/internal/object_storage",
            get(endpoints::object_storage::get_object_handler),
        )
        // Model statistics endpoints
        .route(
            "/internal/models/count",
            get(endpoints::internal::models::count_models_handler),
        )
        .route(
            "/internal/models/usage",
            get(endpoints::internal::models::get_model_usage_handler),
        )
        .route(
            "/internal/models/latency",
            get(endpoints::internal::models::get_model_latency_handler),
        )
        .route(
            "/internal/models/cache_statistics",
            get(endpoints::internal::models::get_cache_statistics_handler),
        )
        // Config snapshot endpoints
        .route(
            "/internal/config",
            get(endpoints::internal::config::get_live_config_handler)
                .post(endpoints::internal::config::write_config_handler),
        )
        .route(
            "/internal/config/{hash}",
            get(endpoints::internal::config::get_config_by_hash_handler),
        )
        // Inference count endpoint
        .route(
            "/internal/inferences/count",
            post(endpoints::internal::count_inferences::count_inferences_handler),
        )
        // Variant statistics endpoint
        .route(
            "/internal/variant_statistics",
            get(endpoints::internal::variant_statistics::get_variant_statistics_handler),
        )
        // Resolve UUID endpoint
        .route(
            "/internal/resolve_uuid/{id}",
            get(endpoints::internal::resolve_uuid::resolve_uuid_handler),
        )
        .route(
            "/internal/synapse/usage_export",
            get(endpoints::internal::synapse::usage_export_handler),
        )
        .route(
            "/internal/synapse/analytics",
            get(endpoints::internal::synapse::analytics_handler),
        )
        .route(
            "/internal/synapse/analysis",
            get(endpoints::internal::synapse_analysis::analysis_handler),
        )
        .route(
            "/internal/synapse/balances",
            get(endpoints::internal::synapse::balances_handler),
        );

    if feature_flags::ENABLE_CONFIG_IN_DATABASE.get() {
        router
            .route(
                "/internal/config_toml",
                get(endpoints::internal::config_toml::get_latest_config_toml_handler),
            )
            // `apply` and `validate` accept the full editable config document in the request
            // body, which can be large — we use `POST` instead of `GET` so callers don't have
            // to URL-encode the entire TOML + referenced file contents into the query string.
            .route(
                "/internal/config_toml/apply",
                post(endpoints::internal::config_toml::apply_config_toml_handler),
            )
            .route(
                "/internal/config_toml/validate",
                post(endpoints::internal::config_toml::validate_config_toml_handler),
            )
    } else {
        router
    }
}

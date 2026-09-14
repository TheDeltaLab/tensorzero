//! External route definitions for the TensorZero Gateway API.
//!
//! This file should remain minimal, containing only endpoint path definitions and their handler mappings.
//! Router construction logic belongs in `router.rs`. This constraint exists because CODEOWNERS
//! requires specific review for route changes.
// All functions here are router builders parameterized on `SwappableAppStateData` by axum's type system.
#![expect(
    clippy::disallowed_types,
    reason = "router builders are parameterized on SwappableAppStateData by axum's type system"
)]

use axum::{
    Router,
    routing::{get, post},
};
use metrics_exporter_prometheus::PrometheusHandle;
use tensorzero_core::endpoints::openai_compatible::build_openai_compatible_routes;
use tensorzero_core::observability::OtelEnabledRoutes;
use tensorzero_core::{endpoints, utils::gateway::SwappableAppStateData};

/// Defines routes that should have top-level OpenTelemetry HTTP spans created
/// All of these routes will have a span named `METHOD <ROUTE>` (e.g. `POST /batch_inference/{batch_id}`)
/// sent to OpenTelemetry
pub fn build_otel_enabled_routes() -> (OtelEnabledRoutes, Router<SwappableAppStateData>) {
    let mut routes = vec![
        ("/inference", post(endpoints::inference::inference_handler)),
        (
            "/batch_inference",
            post(endpoints::batch_inference::start_batch_inference_handler),
        ),
        (
            "/batch_inference/{batch_id}",
            get(endpoints::batch_inference::poll_batch_inference_handler),
        ),
        (
            "/batch_inference/{batch_id}/inference/{inference_id}",
            get(endpoints::batch_inference::poll_batch_inference_handler),
        ),
        ("/feedback", post(endpoints::feedback::feedback_handler)),
    ];
    routes.extend(build_openai_compatible_routes().routes);
    let mut router = Router::new();
    let mut route_names = Vec::with_capacity(routes.len());
    for (path, handler) in routes {
        route_names.push(path);
        router = router.route(path, handler);
    }
    (
        OtelEnabledRoutes {
            routes: route_names,
        },
        router,
    )
}

/// Builds external routes that don't have OpenTelemetry tracing.
pub fn build_non_otel_enabled_routes(
    metrics_handle: PrometheusHandle,
) -> Router<SwappableAppStateData> {
    Router::new()
        .merge(build_observability_routes())
        .merge(build_meta_observability_routes(metrics_handle))
}

/// This function builds the public routes for observability.
///
/// IMPORTANT: Add internal routes to `internal.rs` instead.
fn build_observability_routes() -> Router<SwappableAppStateData> {
    Router::new()
        .route(
            "/v1/inferences/list_inferences",
            post(endpoints::stored_inferences::v1::list_inferences_handler),
        )
        .route(
            "/v1/inferences/get_inferences",
            post(endpoints::stored_inferences::v1::get_inferences_handler),
        )
}

/// This function builds the public routes for meta-observability (e.g. gateway health).
///
/// IMPORTANT: Add internal routes to `internal.rs` instead.
fn build_meta_observability_routes(
    metrics_handle: PrometheusHandle,
) -> Router<SwappableAppStateData> {
    Router::new()
        .route(
            "/metrics",
            get(move || std::future::ready(metrics_handle.render())),
        )
        .route("/status", get(endpoints::status::status_handler))
        .route("/health", get(endpoints::status::health_handler))
}

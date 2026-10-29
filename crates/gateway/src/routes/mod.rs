// Modified by Delta-AI under Apache 2.0
//! Route definitions and endpoint mappings for the TensorZero Gateway API.

mod external;
mod internal;

use axum::Router;
use http::Extensions;
use metrics_exporter_prometheus::PrometheusHandle;
use std::sync::Arc;
use tensorzero_auth::key::TensorZeroAuthError;
use tensorzero_core::error::Error;
use tensorzero_core::observability::{RouterExt as _, TracerWrapper};
#[expect(
    clippy::disallowed_types,
    reason = "router builders are parameterized on SwappableAppStateData by axum's type system"
)]
use tensorzero_core::utils::gateway::SwappableAppStateData;

/// Surfaces `tensorzero_core::error::Error` and `TensorZeroAuthError` from
/// response extensions to the top-level OTLP span as `Status::Error`.
/// Passed into [`RouterExt::apply_top_level_otel_http_trace_layer`] so that
/// `tensorzero-otel` itself doesn't need to know about tensorzero's error
/// types.
fn extract_tensorzero_response_error(ext: &Extensions) -> Option<String> {
    if let Some(error) = ext.get::<Error>() {
        return Some(error.to_string());
    }
    if let Some(error) = ext.get::<TensorZeroAuthError>() {
        return Some(error.to_string());
    }
    None
}

#[expect(
    clippy::disallowed_types,
    reason = "router builders are parameterized on SwappableAppStateData by axum's type system"
)]
pub fn build_api_routes(
    otel_tracer: Option<Arc<TracerWrapper>>,
    metrics_handle: PrometheusHandle,
) -> Router<SwappableAppStateData> {
    let (otel_enabled_routes, otel_enabled_router) = external::build_otel_enabled_routes();
    let router = Router::new()
        .merge(otel_enabled_router)
        // Any routes added here will *not* export any OpenTelemetry spans (since they will not be listed in `otel_enabled_routes`)
        .merge(external::build_non_otel_enabled_routes(metrics_handle))
        .merge(internal::build_internal_non_otel_enabled_routes());
    // Delta 观测规范 §1 边界 request.id（仅 `internal-observability` feature
    // 启用时）：x-request-id 校验/重建/回写 + request_id_invalid_total 计数。
    // 必须加在 otel 层之内（先应用），才能把 request.id 记入顶层 HTTP span；
    // 公开构建（feature 关闭）不引入私有依赖，保持上游行为。
    #[cfg(feature = "internal-observability")]
    let router = router.apply_request_id_layer();
    router.apply_top_level_otel_http_trace_layer(
        otel_tracer,
        otel_enabled_routes,
        extract_tensorzero_response_error,
    )
}

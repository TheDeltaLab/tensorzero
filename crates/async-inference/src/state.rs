// Modified by Delta-AI under Apache 2.0
//! Shared state available to async inference task executions.

use std::time::Duration;

use redis::aio::ConnectionManager;
#[expect(
    clippy::disallowed_types,
    reason = "the async inference worker is gateway-construction code; it needs the swappable state so task executions see the latest config"
)]
use tensorzero_core::utils::gateway::SwappableAppStateData;

/// Application state for the async inference worker.
///
/// Holds the gateway's swappable state (so each task execution loads the
/// latest config) plus the Valkey connection used to relay SSE events.
#[derive(Clone)]
pub struct AsyncInferenceState {
    #[expect(
        clippy::disallowed_types,
        reason = "the async inference worker is gateway-construction code; it needs the swappable state so task executions see the latest config"
    )]
    pub app_state: SwappableAppStateData,
    /// Valkey connection for the per-task SSE event streams.
    pub valkey: ConnectionManager,
    /// TTL applied to each task's Redis event stream (refreshed while the
    /// task is running).
    pub stream_ttl: Duration,
}

// Modified by Delta-AI under Apache 2.0
pub mod cache;
mod rate_limiting;
#[cfg(test)]
mod tests;

use std::time::Duration;

use async_trait::async_trait;
use redis::aio::{ConnectionManager, ConnectionManagerConfig};
use redis::{AsyncCommands, Client, RedisResult};
use tokio::time::timeout;

use crate::db::HealthCheckable;
use crate::error::{DelayedError, ErrorDetails};

/// Response timeout for the dedicated async inference event-stream connection.
///
/// The shared manager keeps the redis-rs default (500ms) so request hot-path
/// users like rate limiting fail fast. Async inference stream commands need
/// more headroom: the initial `XRANGE` replay can straddle a loaded Valkey,
/// and the follow loop's blocking `XREAD` waits up to `XREAD_BLOCK_MS` (5s)
/// server-side before returning empty, so this must comfortably exceed that.
const ASYNC_INFERENCE_STREAM_RESPONSE_TIMEOUT: Duration = Duration::from_secs(10);

/// Connection info for Valkey (Redis-compatible) rate limiting backend.
///
/// Uses `ConnectionManager` which provides:
/// - Automatic reconnection on connection loss
/// - Connection multiplexing for efficient async operations
/// - No connection pool management needed
#[derive(Clone)]
pub enum ValkeyConnectionInfo {
    Enabled {
        connection: Box<ConnectionManager>,
        /// Dedicated manager for the async inference event-stream endpoints
        /// (`GET /v1/async_tasks/{task_id}/stream`), configured with
        /// [`ASYNC_INFERENCE_STREAM_RESPONSE_TIMEOUT`] instead of the redis-rs
        /// 500ms default. `None` for cache-only connections.
        async_inference_stream_connection: Option<Box<ConnectionManager>>,
    },
    Disabled,
}

impl ValkeyConnectionInfo {
    pub async fn new(valkey_url: &str) -> Result<Self, DelayedError> {
        let client = Client::open(valkey_url).map_err(|e| {
            DelayedError::new(ErrorDetails::ValkeyConnection {
                message: format!("Failed to create Valkey client: {e}"),
            })
        })?;

        let mut connection = ConnectionManager::new(client.clone()).await.map_err(|e| {
            DelayedError::new(ErrorDetails::ValkeyConnection {
                message: format!("Failed to connect to Valkey: {e}"),
            })
        })?;

        let async_inference_stream_connection = ConnectionManager::new_with_config(
            client,
            ConnectionManagerConfig::new()
                .set_response_timeout(Some(ASYNC_INFERENCE_STREAM_RESPONSE_TIMEOUT)),
        )
        .await
        .map_err(|e| {
            DelayedError::new(ErrorDetails::ValkeyConnection {
                message: format!("Failed to connect to Valkey: {e}"),
            })
        })?;

        // When creating the connection, load the function library into Valkey.
        Self::load_function_library(&mut connection).await?;

        // Migrate old rate limit keys to new prefixed keys for backwards compatibility.
        Self::migrate_old_ratelimit_keys(&mut connection).await?;

        Ok(Self::Enabled {
            connection: Box::new(connection),
            async_inference_stream_connection: Some(Box::new(async_inference_stream_connection)),
        })
    }

    /// Creates a new connection to Valkey for caching only.
    /// Unlike `new()`, this does NOT load rate limiting Lua functions
    /// or run key migrations, since the cache instance doesn't need them.
    pub async fn new_cache_only(valkey_url: &str) -> Result<Self, DelayedError> {
        let client = Client::open(valkey_url).map_err(|e| {
            DelayedError::new(ErrorDetails::ValkeyConnection {
                message: format!("Failed to create Valkey client: {e}"),
            })
        })?;

        let connection = ConnectionManager::new(client).await.map_err(|e| {
            DelayedError::new(ErrorDetails::ValkeyConnection {
                message: format!("Failed to connect to Valkey: {e}"),
            })
        })?;

        Ok(Self::Enabled {
            connection: Box::new(connection),
            async_inference_stream_connection: None,
        })
    }

    pub fn new_disabled() -> Self {
        Self::Disabled
    }

    pub fn get_connection(&self) -> Option<&ConnectionManager> {
        match self {
            Self::Enabled { connection, .. } => Some(connection),
            Self::Disabled => None,
        }
    }

    /// The dedicated connection for the async inference event-stream
    /// endpoints, with a larger response timeout than the shared manager.
    /// `None` when Valkey is disabled or this is a cache-only connection.
    pub fn get_async_inference_stream_connection(&self) -> Option<&ConnectionManager> {
        match self {
            Self::Enabled {
                async_inference_stream_connection,
                ..
            } => async_inference_stream_connection.as_deref(),
            Self::Disabled => None,
        }
    }

    /// Load the rate limiting function library into Valkey.
    /// This should be called once at startup.
    async fn load_function_library(connection: &mut ConnectionManager) -> Result<(), DelayedError> {
        let lua_code = include_str!("lua/tensorzero_ratelimit.lua");

        // Use FUNCTION LOAD with REPLACE to load/update the library
        let result: RedisResult<()> = redis::cmd("FUNCTION")
            .arg("LOAD")
            .arg("REPLACE")
            .arg(lua_code)
            .query_async(connection)
            .await;
        result.map_err(|e| {
            DelayedError::new(ErrorDetails::ValkeyQuery {
                message: format!("Failed to load function library: {e}"),
            })
        })
    }

    /// Migrate old rate limit keys (`ratelimit:*`) to new prefixed keys (`tensorzero_ratelimit:*`).
    /// This preserves existing rate limit state during upgrades from older versions.
    /// Keys are only copied if the new key doesn't already exist.
    /// The migration runs entirely in Lua for efficiency (single round-trip).
    async fn migrate_old_ratelimit_keys(
        connection: &mut ConnectionManager,
    ) -> Result<(), DelayedError> {
        // Call the Lua function to perform the migration atomically on the server
        let _result: String = redis::cmd("FCALL")
            .arg("tensorzero_migrate_old_keys_v1")
            .arg(0) // No keys passed
            .query_async(connection)
            .await
            .map_err(|e| {
                DelayedError::new(ErrorDetails::ValkeyQuery {
                    message: format!("Failed to migrate old rate limit keys: {e}"),
                })
            })?;

        Ok(())
    }
}

const HEALTH_CHECK_TIMEOUT_MS: u64 = 1000;

#[async_trait]
impl HealthCheckable for ValkeyConnectionInfo {
    async fn health(&self) -> Result<(), DelayedError> {
        match self {
            Self::Disabled => Ok(()),
            Self::Enabled { connection, .. } => {
                let check = async {
                    let mut conn = connection.clone();
                    let _: String = conn.ping().await.map_err(|e| {
                        DelayedError::new(ErrorDetails::ValkeyConnection {
                            message: format!("Valkey health check failed: {e}"),
                        })
                    })?;
                    Ok(())
                };

                match timeout(Duration::from_millis(HEALTH_CHECK_TIMEOUT_MS), check).await {
                    Ok(Ok(())) => Ok(()),
                    Ok(Err(e)) => Err(e),
                    Err(_) => Err(DelayedError::new(ErrorDetails::ValkeyConnection {
                        message: "Valkey health check timed out".to_string(),
                    })),
                }
            }
        }
    }
}

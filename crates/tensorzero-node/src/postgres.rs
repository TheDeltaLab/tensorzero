// Modified by Delta-AI under Apache 2.0
use secrecy::ExposeSecret;
use tensorzero_auth::{
    constants::{DEFAULT_ORGANIZATION, DEFAULT_WORKSPACE},
    key::TensorZeroApiKey,
    postgres::ApiKeyValidationResult,
};
use tensorzero_core::{
    config::Config, db::postgres::PostgresConnectionInfo, utils::gateway::setup_postgres,
};

#[napi(js_name = "PostgresClient")]
pub struct PostgresClient {
    connection_info: PostgresConnectionInfo,
}

#[napi]
impl PostgresClient {
    #[napi(factory)]
    pub async fn from_postgres_url(postgres_url: String) -> Result<Self, napi::Error> {
        // Create a minimal config just for postgres connection pool size
        // The default pool size is 10 which should be reasonable
        let config = Config::new_empty()
            .await
            .map_err(|e| napi::Error::from_reason(format!("Failed to setup Postgres: {e}")))?
            .dangerous_into_config_without_writing();

        let connection_info = setup_postgres(&config, Some(&postgres_url))
            .await
            .map_err(|e| napi::Error::from_reason(format!("Failed to setup Postgres: {e}")))?;

        Ok(Self { connection_info })
    }

    #[napi]
    pub async fn create_api_key(
        &self,
        description: Option<String>,
        expires_at: Option<String>,
    ) -> Result<String, napi::Error> {
        let pool = self
            .connection_info
            .get_pool()
            .ok_or_else(|| napi::Error::from_reason("Postgres connection not available"))?;

        let expires_at = expires_at
            .as_deref()
            .map(tensorzero_auth::postgres::parse_expires_at)
            .transpose()
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;

        let key = tensorzero_auth::postgres::create_key(
            DEFAULT_ORGANIZATION,
            DEFAULT_WORKSPACE,
            description.as_deref(),
            expires_at,
            pool,
        )
        .await
        .map_err(|e| napi::Error::from_reason(format!("Failed to create API key: {e}")))?;

        Ok(key.expose_secret().to_string())
    }

    #[napi]
    pub async fn list_api_keys(
        &self,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<String, napi::Error> {
        let pool = self
            .connection_info
            .get_pool()
            .ok_or_else(|| napi::Error::from_reason("Postgres connection not available"))?;

        let keys = tensorzero_auth::postgres::list_key_info(None, None, limit, offset, pool)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Failed to list API keys: {e}")))?;

        let public_ids: Vec<String> = keys.iter().map(|key| key.public_id.clone()).collect();
        let last_used =
            tensorzero_core::db::postgres::api_key_usage::last_used_at_by_api_key_public_ids(
                pool,
                &public_ids,
            )
            .await
            .unwrap_or_default();

        let mut encoded = serde_json::to_value(&keys).map_err(|e| {
            napi::Error::from_reason(format!("Failed to serialize API keys list: {e}"))
        })?;
        if let Some(arr) = encoded.as_array_mut() {
            for item in arr {
                let Some(obj) = item.as_object_mut() else {
                    continue;
                };
                let Some(public_id) = obj.get("public_id").and_then(|v| v.as_str()) else {
                    continue;
                };
                if let Some(ts) = last_used.get(public_id) {
                    obj.insert(
                        "last_used_at".to_string(),
                        serde_json::Value::String(ts.clone()),
                    );
                }
            }
        }

        serde_json::to_string(&encoded).map_err(|e| {
            napi::Error::from_reason(format!("Failed to serialize API keys list: {e}"))
        })
    }

    #[napi]
    pub async fn disable_api_key(&self, public_id: String) -> Result<String, napi::Error> {
        let pool = self
            .connection_info
            .get_pool()
            .ok_or_else(|| napi::Error::from_reason("Postgres connection not available"))?;

        let disabled_at = tensorzero_auth::postgres::disable_key(&public_id, pool)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Failed to disable API key: {e}")))?;

        serde_json::to_string(&disabled_at).map_err(|e| {
            napi::Error::from_reason(format!("Failed to serialize disabled_at timestamp: {e}"))
        })
    }

    /// Validates an API key against the `tensorzero_auth_api_key` table, reusing the same
    /// parsing and lookup that the gateway's auth middleware uses
    /// (`tensorzero_auth::postgres::check_key`).
    ///
    /// Returns a JSON-encoded `ApiKeyValidationResult` describing the auth outcome (valid,
    /// invalid format, missing, disabled, or expired). Throws `napi::Error` only for
    /// infrastructure failures (Postgres unavailable, query errors, serialization failures)
    /// — callers must therefore distinguish auth failures (parsed `type != "valid"`) from
    /// infra failures (thrown error); only the former should map to a 401.
    #[napi]
    pub async fn validate_api_key(&self, key: String) -> Result<String, napi::Error> {
        let pool = self
            .connection_info
            .get_pool()
            .ok_or_else(|| napi::Error::from_reason("Postgres connection not available"))?;

        let result = match TensorZeroApiKey::parse(&key) {
            Ok(parsed_key) => tensorzero_auth::postgres::check_key(&parsed_key, pool)
                .await
                .map_err(|e| napi::Error::from_reason(format!("Failed to validate API key: {e}")))?
                .into(),
            Err(_) => ApiKeyValidationResult::InvalidFormat,
        };

        serde_json::to_string(&result).map_err(|e| {
            napi::Error::from_reason(format!("Failed to serialize validation result: {e}"))
        })
    }

    #[napi]
    pub async fn update_api_key_description(
        &self,
        public_id: String,
        description: Option<String>,
    ) -> Result<String, napi::Error> {
        let pool = self
            .connection_info
            .get_pool()
            .ok_or_else(|| napi::Error::from_reason("Postgres connection not available"))?;

        let key = tensorzero_auth::postgres::update_key_description(
            &public_id,
            description.as_deref(),
            pool,
        )
        .await
        .map_err(|e| {
            napi::Error::from_reason(format!("Failed to update API key description: {e}"))
        })?;

        serde_json::to_string(&key).map_err(|e| {
            napi::Error::from_reason(format!("Failed to serialize updated API key: {e}"))
        })
    }
}

// Modified by Delta-AI under Apache 2.0
use std::collections::HashSet;

use chrono::SubsecRound;
use chrono::{DateTime, Utc};
use futures::TryStreamExt;
use secrecy::{ExposeSecret, SecretString};
use serde::Serialize;
use sha2::{Digest, Sha256};
use sqlx::PgPool;
use sqlx::Row;

use crate::key::{TensorZeroApiKey, TensorZeroAuthError, secure_fresh_api_key};

pub fn make_migrator() -> sqlx::migrate::Migrator {
    sqlx::migrate!("src/postgres/migrations")
}

pub struct MigrationsData {
    pub applied: HashSet<i64>,
    pub expected: HashSet<i64>,
}

/// Helper function to retrieve the set of applied migrations from the database.
/// We pull this out so that the error can be mapped in one place.
/// This is almost the same as the corresponding `get_applied_migrations` function in 'tensorzero_core', but with a different table name.
async fn get_applied_migrations(pool: &PgPool) -> Result<HashSet<i64>, sqlx::Error> {
    let mut applied_migrations: HashSet<i64> = HashSet::new();
    let mut rows =
        sqlx::query("SELECT version FROM tensorzero_auth__sqlx_migrations WHERE success = true ORDER BY version")
            .fetch(pool);
    while let Some(row) = rows.try_next().await? {
        let id: i64 = row.try_get("version")?;
        applied_migrations.insert(id);
    }
    Ok(applied_migrations)
}

pub async fn get_migrations_data(pool: &PgPool) -> Result<MigrationsData, sqlx::Error> {
    let migrator = make_migrator();
    let expected_migrations: HashSet<i64> = migrator.iter().map(|m| m.version).collect();
    // Query the database for all successfully applied migration versions.
    let applied_migrations = get_applied_migrations(pool).await?;
    Ok(MigrationsData {
        applied: applied_migrations,
        expected: expected_migrations,
    })
}

/// Create a new API key, and store in the database.
/// Returns the generated API key
pub async fn create_key(
    organization: &str,
    workspace: &str,
    description: Option<&str>,
    expires_at: Option<DateTime<Utc>>,
    pool: &PgPool,
) -> Result<SecretString, TensorZeroAuthError> {
    let key = secure_fresh_api_key();
    let parsed_key = TensorZeroApiKey::parse(key.expose_secret())?;
    sqlx::query!(
        "INSERT INTO tensorzero_auth_api_key (organization, workspace, description, public_id, hash, expires_at) VALUES ($1, $2, $3, $4, $5, $6)",
        organization,
        workspace,
        description,
        parsed_key.public_id,
        parsed_key.hashed_long_key.expose_secret(),
        expires_at
    )
    .execute(pool)
    .await?;
    Ok(key)
}

pub fn parse_expires_at(s: &str) -> Result<DateTime<Utc>, TensorZeroAuthError> {
    DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| TensorZeroAuthError::InvalidExpiresAt(e.to_string()))
}

#[derive(Debug, Clone)]
pub enum AuthResult {
    /// The API key exists and is not disabled.
    Success(KeyInfo),
    /// The API key exists, but was disabled at the specified time.
    Disabled(DateTime<Utc>, KeyInfo),
    /// The API key exists, but expired at the specified time.
    Expired(DateTime<Utc>, KeyInfo),
    /// The API key does not exist.
    MissingKey,
}

/// Result of validating an API key, suitable for serialization across language boundaries
/// (NAPI → TypeScript). Auth-related outcomes are encoded as variants here so that
/// callers can distinguish them from infrastructure failures (which propagate as thrown
/// errors at the NAPI layer).
#[derive(ts_rs::TS, Debug, Clone, Serialize)]
#[ts(export)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ApiKeyValidationResult {
    /// The API key parsed and matches an active, non-expired row.
    Success { key_info: KeyInfo },
    /// The provided string did not parse as a TensorZero API key.
    InvalidFormat,
    /// The provided key parsed but does not exist in the database.
    Missing,
    /// The provided key exists but has been disabled.
    Disabled,
    /// The provided key exists but is past its expiration.
    Expired,
}

impl From<AuthResult> for ApiKeyValidationResult {
    fn from(result: AuthResult) -> Self {
        match result {
            AuthResult::Success(key_info) => Self::Success { key_info },
            AuthResult::Disabled(_, _) => Self::Disabled,
            AuthResult::Expired(_, _) => Self::Expired,
            AuthResult::MissingKey => Self::Missing,
        }
    }
}

#[derive(ts_rs::TS, sqlx::FromRow, Debug, PartialEq, Eq, Clone, Serialize)]
#[ts(export)]
pub struct KeyInfo {
    pub public_id: String,
    pub organization: String,
    pub workspace: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub description: Option<String>,
    pub created_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub disabled_at: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub expires_at: Option<DateTime<Utc>>,
}

/// Looks up an API key in the database, and checks that it was not disabled or expired.
pub async fn check_key(
    key: &TensorZeroApiKey,
    pool: &PgPool,
) -> Result<AuthResult, TensorZeroAuthError> {
    let key = sqlx::query_as!(
        KeyInfo,
        "SELECT public_id, organization, workspace, description, created_at, disabled_at, expires_at from tensorzero_auth_api_key WHERE public_id = $1 AND hash = $2",
        key.public_id,
        key.hashed_long_key.expose_secret()
    ).fetch_optional(pool).await?;
    match key {
        Some(key) => {
            if let Some(disabled_at) = key.disabled_at {
                Ok(AuthResult::Disabled(disabled_at, key))
            } else if let Some(expires_at) = key.expires_at {
                if expires_at <= Utc::now() {
                    Ok(AuthResult::Expired(expires_at, key))
                } else {
                    Ok(AuthResult::Success(key))
                }
            } else {
                Ok(AuthResult::Success(key))
            }
        }
        None => Ok(AuthResult::MissingKey),
    }
}

#[derive(sqlx::FromRow)]
struct SynapseKeyRow {
    public_id: String,
    organization: String,
    workspace: String,
    description: Option<String>,
    created_at: DateTime<Utc>,
    disabled_at: Option<DateTime<Utc>>,
    expires_at: Option<DateTime<Utc>>,
    bcrypt_hash: String,
}

async fn insert_synapse_api_key(
    organization: &str,
    workspace: &str,
    description: Option<&str>,
    public_id: &str,
    bcrypt_hash: &str,
    pool: &PgPool,
) -> Result<(), TensorZeroAuthError> {
    sqlx::query(
        "INSERT INTO tensorzero_auth_synapse_api_key \
         (organization, workspace, description, public_id, bcrypt_hash) \
         VALUES ($1, $2, $3, $4, $5)",
    )
    .bind(organization)
    .bind(workspace)
    .bind(description)
    .bind(public_id)
    .bind(bcrypt_hash)
    .execute(pool)
    .await?;
    Ok(())
}

/// Import a precomputed Synapse bcrypt hash (cost 10). Does not re-hash.
///
/// The stored `public_id` is derived from the bcrypt string. On first successful
/// authentication, [`align_synapse_key_public_id`] rewrites it to the plaintext
/// tag id used on inferences.
pub async fn import_synapse_key(
    organization: &str,
    workspace: &str,
    description: Option<&str>,
    bcrypt_hash: &str,
    pool: &PgPool,
) -> Result<String, TensorZeroAuthError> {
    let digest = hex::encode(Sha256::digest(bcrypt_hash.as_bytes()));
    let public_id = format!("syn{}", &digest[..9]);
    insert_synapse_api_key(
        organization,
        workspace,
        description,
        &public_id,
        bcrypt_hash,
        pool,
    )
    .await?;
    Ok(public_id)
}

const SYNAPSE_BCRYPT_COST: u32 = 10;

/// Import either a bcrypt hash (`$2...`) or a plaintext `sk-syn-v1-...` key.
pub async fn import_synapse_key_or_plaintext(
    organization: &str,
    workspace: &str,
    description: Option<&str>,
    hash_or_key: &str,
    pool: &PgPool,
) -> Result<String, TensorZeroAuthError> {
    if hash_or_key.starts_with("$2") {
        return import_synapse_key(organization, workspace, description, hash_or_key, pool).await;
    }
    if !TensorZeroApiKey::is_synapse_key(hash_or_key) {
        return Err(TensorZeroAuthError::InvalidKeyFormat(
            "expected a bcrypt hash or a plaintext sk-syn-v1- key",
        ));
    }
    let public_id = TensorZeroApiKey::from_synapse_plaintext(hash_or_key)
        .get_public_id()
        .to_string();
    let bcrypt_hash = bcrypt::hash(hash_or_key, SYNAPSE_BCRYPT_COST).map_err(|e| {
        TensorZeroAuthError::Middleware {
            message: format!("Failed to bcrypt Synapse key: {e}"),
            key_info: None,
        }
    })?;
    insert_synapse_api_key(
        organization,
        workspace,
        description,
        &public_id,
        &bcrypt_hash,
        pool,
    )
    .await?;
    Ok(public_id)
}

/// Rewrite an imported Synapse `public_id` so it matches the id stamped on
/// inference tags (`syn` + first 9 hex chars of SHA-256(plaintext)).
///
/// Older imports hashed the bcrypt string instead, so the API-keys dropdown
/// value never matched `tensorzero::api_key_public_id`.
pub async fn align_synapse_key_public_id(
    tagged_public_id: &str,
    mut key: KeyInfo,
    pool: &PgPool,
) -> Result<KeyInfo, TensorZeroAuthError> {
    key.public_id = key.public_id.trim().to_string();
    if key.public_id == tagged_public_id {
        return Ok(key);
    }
    let stored_public_id = key.public_id.clone();
    let result = sqlx::query(
        "UPDATE tensorzero_auth_synapse_api_key \
         SET public_id = $1, updated_at = CURRENT_TIMESTAMP \
         WHERE btrim(public_id::text) = $2 \
           AND NOT EXISTS ( \
             SELECT 1 FROM tensorzero_auth_synapse_api_key other \
             WHERE btrim(other.public_id::text) = $1 \
           )",
    )
    .bind(tagged_public_id)
    .bind(&stored_public_id)
    .execute(pool)
    .await?;
    if result.rows_affected() > 0 {
        key.public_id = tagged_public_id.to_string();
    }
    Ok(key)
}

/// Scan imported Synapse bcrypt hashes. Matches Synapse's compare-all-keys lookup.
pub async fn check_synapse_key(
    plaintext: &str,
    pool: &PgPool,
) -> Result<AuthResult, TensorZeroAuthError> {
    let tagged_public_id = TensorZeroApiKey::from_synapse_plaintext(plaintext)
        .get_public_id()
        .to_string();
    let rows: Vec<SynapseKeyRow> = sqlx::query_as(
        "SELECT btrim(public_id::text) AS public_id, organization, workspace, description, created_at, disabled_at, expires_at, bcrypt_hash \
         FROM tensorzero_auth_synapse_api_key",
    )
    .fetch_all(pool)
    .await?;
    for row in rows {
        let ok = bcrypt::verify(plaintext, &row.bcrypt_hash).unwrap_or(false);
        if !ok {
            continue;
        }
        let key = align_synapse_key_public_id(
            &tagged_public_id,
            KeyInfo {
                public_id: row.public_id,
                organization: row.organization,
                workspace: row.workspace,
                description: row.description,
                created_at: row.created_at,
                disabled_at: row.disabled_at,
                expires_at: row.expires_at,
            },
            pool,
        )
        .await?;
        if let Some(disabled_at) = key.disabled_at {
            return Ok(AuthResult::Disabled(disabled_at, key));
        }
        if let Some(expires_at) = key.expires_at
            && expires_at <= Utc::now()
        {
            return Ok(AuthResult::Expired(expires_at, key));
        }
        return Ok(AuthResult::Success(key));
    }
    Ok(AuthResult::MissingKey)
}

/// Marks an API key as disabled in the database by its public_id
/// Returns the `disabled_at` timestamp that was set in the database.
pub async fn disable_key(
    public_id: &str,
    pool: &PgPool,
) -> Result<DateTime<Utc>, TensorZeroAuthError> {
    // Round to microseconds, since postgres only has microsecond precision
    // This ensures that the value we return matches the value we set in the database.
    let now = Utc::now().round_subsecs(6);
    let native = sqlx::query!(
        "UPDATE tensorzero_auth_api_key SET disabled_at = $1, updated_at = $1 WHERE public_id = $2",
        now,
        public_id
    )
    .execute(pool)
    .await?;
    if native.rows_affected() == 0 {
        sqlx::query(
            "UPDATE tensorzero_auth_synapse_api_key SET disabled_at = $1, updated_at = $1 WHERE btrim(public_id::text) = $2",
        )
        .bind(now)
        .bind(public_id)
        .execute(pool)
        .await?;
    }
    Ok(now)
}

/// Updates the description for the API key with the given public_id
pub async fn update_key_description(
    public_id: &str,
    description: Option<&str>,
    pool: &PgPool,
) -> Result<KeyInfo, TensorZeroAuthError> {
    if let Some(key) = sqlx::query_as::<_, KeyInfo>(
        "UPDATE tensorzero_auth_api_key
           SET description = $1, updated_at = NOW()
           WHERE public_id = $2
           RETURNING public_id, organization, workspace, description, created_at, disabled_at, expires_at",
    )
    .bind(description)
    .bind(public_id)
    .fetch_optional(pool)
    .await?
    {
        return Ok(key);
    }

    let key = sqlx::query_as::<_, KeyInfo>(
        "UPDATE tensorzero_auth_synapse_api_key
           SET description = $1, updated_at = NOW()
           WHERE btrim(public_id::text) = $2
           RETURNING btrim(public_id::text) AS public_id, organization, workspace, description, created_at, disabled_at, expires_at",
    )
    .bind(description)
    .bind(public_id)
    .fetch_one(pool)
    .await?;

    Ok(key)
}

/// Fetches metadata for a single API key by its `public_id`.
/// Returns `Ok(None)` if no key with that `public_id` exists.
pub async fn get_key_info(
    public_id: &str,
    pool: &PgPool,
) -> Result<Option<KeyInfo>, TensorZeroAuthError> {
    if let Some(key) = sqlx::query_as::<_, KeyInfo>(
        "SELECT public_id, organization, workspace, description, created_at, disabled_at, expires_at \
         FROM tensorzero_auth_api_key \
         WHERE public_id = $1",
    )
    .bind(public_id)
    .fetch_optional(pool)
    .await?
    {
        return Ok(Some(key));
    }

    let key = sqlx::query_as::<_, KeyInfo>(
        "SELECT btrim(public_id::text) AS public_id, organization, workspace, description, created_at, disabled_at, expires_at FROM tensorzero_auth_synapse_api_key WHERE btrim(public_id::text) = $1",
    )
    .bind(public_id)
    .fetch_optional(pool)
    .await?;
    Ok(key)
}

fn paginate_keys(mut keys: Vec<KeyInfo>, limit: Option<u32>, offset: Option<u32>) -> Vec<KeyInfo> {
    keys.sort_by(|left, right| {
        right
            .created_at
            .cmp(&left.created_at)
            .then_with(|| left.public_id.cmp(&right.public_id))
    });
    let offset = offset.unwrap_or(0) as usize;
    keys.into_iter()
        .skip(offset)
        .take(limit.map_or(usize::MAX, |limit| limit as usize))
        .collect()
}

/// Lists all API keys in the database, optionally filtered by organization
/// and/or workspace, with an optional limit and offset.
///
/// Includes imported Synapse keys (`tensorzero_auth_synapse_api_key`) alongside
/// native TensorZero keys. Workspace names are not unique across organizations,
/// so the `workspace` filter requires an `organization` filter to be set.
pub async fn list_key_info(
    organization: Option<String>,
    workspace: Option<String>,
    limit: Option<u32>,
    offset: Option<u32>,
    pool: &PgPool,
) -> Result<Vec<KeyInfo>, TensorZeroAuthError> {
    if workspace.is_some() && organization.is_none() {
        return Err(TensorZeroAuthError::WorkspaceFilterRequiresOrganization);
    }
    let native = sqlx::query_as::<_, KeyInfo>(
        "SELECT public_id, organization, workspace, description, created_at, disabled_at, expires_at FROM tensorzero_auth_api_key WHERE (organization = $1 OR $1 is NULL) AND (workspace = $2 OR $2 is NULL)",
    )
    .bind(&organization)
    .bind(&workspace)
    .fetch_all(pool)
    .await?;
    let synapse = sqlx::query_as::<_, KeyInfo>(
        "SELECT btrim(public_id::text) AS public_id, organization, workspace, description, created_at, disabled_at, expires_at FROM tensorzero_auth_synapse_api_key WHERE (organization = $1 OR $1 is NULL) AND (workspace = $2 OR $2 is NULL)",
    )
    .bind(&organization)
    .bind(&workspace)
    .fetch_all(pool)
    .await?;

    let mut keys = native;
    keys.extend(synapse);
    Ok(paginate_keys(keys, limit, offset))
}

#[cfg(test)]
mod tests {
    use super::*;
    use googletest::prelude::*;

    fn sample_key(public_id: &str, created_at: DateTime<Utc>) -> KeyInfo {
        KeyInfo {
            public_id: public_id.to_string(),
            organization: "org".to_string(),
            workspace: "ws".to_string(),
            description: None,
            created_at,
            disabled_at: None,
            expires_at: None,
        }
    }

    #[gtest]
    fn paginate_keys_orders_newest_first() {
        let older = Utc::now();
        let newer = older + chrono::TimeDelta::seconds(1);
        let keys = paginate_keys(
            vec![
                sample_key("nativeaaaaaa", older),
                sample_key("synbbbbbbbb", newer),
            ],
            None,
            None,
        );
        expect_that!(keys.len(), eq(2));
        expect_that!(keys[0].public_id.as_str(), eq("synbbbbbbbb"));
        expect_that!(keys[1].public_id.as_str(), eq("nativeaaaaaa"));
    }

    #[gtest]
    fn paginate_keys_applies_limit_and_offset() {
        let t0 = Utc::now();
        let t1 = t0 + chrono::TimeDelta::seconds(1);
        let t2 = t0 + chrono::TimeDelta::seconds(2);
        let keys = paginate_keys(
            vec![
                sample_key("key0aaaaaaaa", t0),
                sample_key("key1aaaaaaaa", t1),
                sample_key("key2aaaaaaaa", t2),
            ],
            Some(1),
            Some(1),
        );
        expect_that!(keys.len(), eq(1));
        expect_that!(keys[0].public_id.as_str(), eq("key1aaaaaaaa"));
    }
}

// Modified by Delta-AI under Apache 2.0
//! Postgres queries for per-inference protection from retention cleanup.
//!
//! Protected inferences (`protected_at` set on the metadata row) are archived
//! into the non-partitioned `*_archive` tables before retention drops old
//! partitions, so they survive retention forever.

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::db::query_helpers::uuid_to_datetime;
use crate::error::{Error, ErrorDetails};

use super::PostgresConnectionInfo;

/// Protection state of a single inference.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct InferenceProtectionRow {
    pub id: Uuid,
    pub function_type: String,
    pub protected_at: DateTime<Utc>,
}

impl PostgresConnectionInfo {
    /// Sets (or clears) the `protected_at` flag on an inference, looking in
    /// the live tables first and then in the archive tables.
    ///
    /// Returns the function type (`chat` or `json`) and the resulting
    /// `protected_at` value. Clearing protection on an archived inference is
    /// not supported (archived rows are kept forever); protecting an archived
    /// inference is a no-op success since archived rows are already protected.
    ///
    /// `created_at` is derived from the UUIDv7 so the `created_at = $3`
    /// predicate keeps partition pruning effective.
    pub async fn set_inference_protection(
        &self,
        inference_id: Uuid,
        protected: bool,
    ) -> Result<(String, Option<DateTime<Utc>>), Error> {
        let pool = self.get_pool_result().map_err(|e| e.log())?;
        let created_at = uuid_to_datetime(inference_id)?;

        // Live tables: chat first, then json.
        if let Some(row) = sqlx::query!(
            r"
            UPDATE tensorzero.chat_inferences
            SET protected_at = CASE WHEN $2 THEN NOW() ELSE NULL END
            WHERE id = $1 AND created_at = $3
            RETURNING protected_at
            ",
            inference_id,
            protected,
            created_at,
        )
        .fetch_optional(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to update protection state in `chat_inferences`: {e}"),
            })
        })? {
            return Ok(("chat".to_string(), row.protected_at));
        }

        if let Some(row) = sqlx::query!(
            r"
            UPDATE tensorzero.json_inferences
            SET protected_at = CASE WHEN $2 THEN NOW() ELSE NULL END
            WHERE id = $1 AND created_at = $3
            RETURNING protected_at
            ",
            inference_id,
            protected,
            created_at,
        )
        .fetch_optional(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to update protection state in `json_inferences`: {e}"),
            })
        })? {
            return Ok(("json".to_string(), row.protected_at));
        }

        // Archive tables: rows are already protected forever.
        if let Some(protected_at) = get_archived_protection(
            pool,
            "tensorzero.chat_inferences_archive",
            inference_id,
            created_at,
        )
        .await?
        {
            return archived_protection_result(inference_id, "chat", protected_at, protected);
        }
        if let Some(protected_at) = get_archived_protection(
            pool,
            "tensorzero.json_inferences_archive",
            inference_id,
            created_at,
        )
        .await?
        {
            return archived_protection_result(inference_id, "json", protected_at, protected);
        }

        Err(Error::new(ErrorDetails::InferenceNotFound { inference_id }))
    }

    /// Returns the protection state for the given inference IDs (only ids
    /// with `protected_at` set are returned), across both the live tables and
    /// the archive tables.
    pub async fn get_inferences_protection(
        &self,
        ids: &[Uuid],
    ) -> Result<Vec<InferenceProtectionRow>, Error> {
        let pool = self.get_pool_result().map_err(|e| e.log())?;

        let mut protection = sqlx::query_as!(
            InferenceProtectionRow,
            r#"
            SELECT id AS "id!", 'chat'::text AS "function_type!", protected_at AS "protected_at!"
            FROM tensorzero.chat_inferences
            WHERE id = ANY($1) AND protected_at IS NOT NULL
            UNION ALL
            SELECT id AS "id!", 'chat'::text AS "function_type!", protected_at AS "protected_at!"
            FROM tensorzero.chat_inferences_archive
            WHERE id = ANY($1) AND protected_at IS NOT NULL
            "#,
            ids,
        )
        .fetch_all(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to query chat inference protection state: {e}"),
            })
        })?;

        let json_protection = sqlx::query_as!(
            InferenceProtectionRow,
            r#"
            SELECT id AS "id!", 'json'::text AS "function_type!", protected_at AS "protected_at!"
            FROM tensorzero.json_inferences
            WHERE id = ANY($1) AND protected_at IS NOT NULL
            UNION ALL
            SELECT id AS "id!", 'json'::text AS "function_type!", protected_at AS "protected_at!"
            FROM tensorzero.json_inferences_archive
            WHERE id = ANY($1) AND protected_at IS NOT NULL
            "#,
            ids,
        )
        .fetch_all(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to query json inference protection state: {e}"),
            })
        })?;

        protection.extend(json_protection);
        Ok(protection)
    }
}

/// Reads `protected_at` from one archive table (chat or json variant of the
/// same static query shape). Returns `Some(protected_at)` if the row exists.
async fn get_archived_protection(
    pool: &sqlx::PgPool,
    table: &str,
    inference_id: Uuid,
    created_at: DateTime<Utc>,
) -> Result<Option<DateTime<Utc>>, Error> {
    // The archive table name is fixed by the caller; both archive tables have
    // an identical `(id, created_at) -> protected_at` shape.
    let protected_at: Option<DateTime<Utc>> = match table {
        "tensorzero.chat_inferences_archive" => sqlx::query_scalar!(
            r#"
            SELECT protected_at AS "protected_at!"
            FROM tensorzero.chat_inferences_archive
            WHERE id = $1 AND created_at = $2
            "#,
            inference_id,
            created_at,
        )
        .fetch_optional(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to read protection state from `{table}`: {e}"),
            })
        })?,
        "tensorzero.json_inferences_archive" => sqlx::query_scalar!(
            r#"
            SELECT protected_at AS "protected_at!"
            FROM tensorzero.json_inferences_archive
            WHERE id = $1 AND created_at = $2
            "#,
            inference_id,
            created_at,
        )
        .fetch_optional(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to read protection state from `{table}`: {e}"),
            })
        })?,
        _ => {
            return Err(Error::new(ErrorDetails::PostgresQuery {
                message: format!("Unknown archive table `{table}`"),
            }));
        }
    };
    Ok(protected_at)
}

fn archived_protection_result(
    inference_id: Uuid,
    function_type: &str,
    protected_at: DateTime<Utc>,
    protected: bool,
) -> Result<(String, Option<DateTime<Utc>>), Error> {
    if protected {
        // Archived rows are already protected forever: no-op success.
        Ok((function_type.to_string(), Some(protected_at)))
    } else {
        Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!(
                "Cannot unprotect inference `{inference_id}`: it has been archived and archived inferences are kept forever"
            ),
        }))
    }
}

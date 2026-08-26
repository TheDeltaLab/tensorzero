// Modified by Delta-AI under Apache 2.0
//! Last-used timestamps for API keys, derived from inference tags.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use sqlx::PgPool;

use crate::error::{Error, ErrorDetails};

#[derive(sqlx::FromRow)]
struct LastUsedRow {
    public_id: String,
    last_used_at: DateTime<Utc>,
}

/// `MAX(created_at)` per `tensorzero::api_key_public_id` tag for the given ids.
///
/// Uses the GIN `tags` indexes (`?` / `->>`) rather than a full-table `COUNT(*)`.
pub async fn last_used_at_by_api_key_public_ids(
    pool: &PgPool,
    public_ids: &[String],
) -> Result<HashMap<String, String>, Error> {
    if public_ids.is_empty() {
        return Ok(HashMap::new());
    }
    let rows: Vec<LastUsedRow> = sqlx::query_as(
        r"
        SELECT tags->>'tensorzero::api_key_public_id' AS public_id,
               MAX(created_at) AS last_used_at
        FROM (
            SELECT tags, created_at
            FROM tensorzero.chat_inferences
            WHERE tags ? 'tensorzero::api_key_public_id'
              AND tags->>'tensorzero::api_key_public_id' = ANY($1)
            UNION ALL
            SELECT tags, created_at
            FROM tensorzero.json_inferences
            WHERE tags ? 'tensorzero::api_key_public_id'
              AND tags->>'tensorzero::api_key_public_id' = ANY($1)
        ) usage
        GROUP BY 1
        ",
    )
    .bind(public_ids)
    .fetch_all(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to load API key last-used times: {e}"),
        })
    })?;
    Ok(rows
        .into_iter()
        .filter(|row| !row.public_id.is_empty())
        .map(|row| (row.public_id, row.last_used_at.to_rfc3339()))
        .collect())
}

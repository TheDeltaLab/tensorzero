// Modified by Delta-AI under Apache 2.0
//! Postgres queries for Azure dashboard allowlist users.

use chrono::{DateTime, Utc};
use serde::Serialize;
use sqlx::PgPool;

use crate::error::{Error, ErrorDetails};

const BOOTSTRAP_CREATED_BY: &str = "toml:gateway.ui.admin_emails";

#[derive(Clone, Debug, Serialize, sqlx::FromRow)]
pub struct DashboardUserRow {
    pub email: String,
    pub is_admin: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_by: Option<String>,
}

pub async fn seed_bootstrap_admins(pool: &PgPool, emails: &[String]) -> Result<(), Error> {
    for email in emails {
        sqlx::query(
            r"
            INSERT INTO tensorzero.dashboard_users (email, is_admin, created_by)
            VALUES ($1, TRUE, $2)
            ON CONFLICT (email) DO UPDATE
            SET is_admin = TRUE,
                updated_at = NOW()
            ",
        )
        .bind(email)
        .bind(BOOTSTRAP_CREATED_BY)
        .execute(pool)
        .await?;
    }
    Ok(())
}

pub async fn get_dashboard_user(
    pool: &PgPool,
    email: &str,
) -> Result<Option<DashboardUserRow>, Error> {
    Ok(sqlx::query_as(
        r"
        SELECT email, is_admin, created_at, updated_at, created_by
        FROM tensorzero.dashboard_users
        WHERE email = $1
        ",
    )
    .bind(email)
    .fetch_optional(pool)
    .await?)
}

pub async fn list_dashboard_users(pool: &PgPool) -> Result<Vec<DashboardUserRow>, Error> {
    Ok(sqlx::query_as(
        r"
        SELECT email, is_admin, created_at, updated_at, created_by
        FROM tensorzero.dashboard_users
        ORDER BY is_admin DESC, email ASC
        ",
    )
    .fetch_all(pool)
    .await?)
}

pub fn is_unique_violation(err: &sqlx::Error) -> bool {
    err.as_database_error()
        .and_then(|e| e.code())
        .is_some_and(|code| code == "23505")
}

pub async fn insert_dashboard_user(
    pool: &PgPool,
    email: &str,
    is_admin: bool,
    created_by: &str,
) -> Result<DashboardUserRow, Error> {
    match sqlx::query_as(
        r"
        INSERT INTO tensorzero.dashboard_users (email, is_admin, created_by)
        VALUES ($1, $2, $3)
        RETURNING email, is_admin, created_at, updated_at, created_by
        ",
    )
    .bind(email)
    .bind(is_admin)
    .bind(created_by)
    .fetch_one(pool)
    .await
    {
        Ok(row) => Ok(row),
        Err(err) if is_unique_violation(&err) => Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!("Dashboard user `{email}` already exists"),
        })),
        Err(err) => Err(err.into()),
    }
}

pub async fn set_dashboard_user_admin(
    pool: &PgPool,
    email: &str,
    is_admin: bool,
) -> Result<Option<DashboardUserRow>, Error> {
    Ok(sqlx::query_as(
        r"
        UPDATE tensorzero.dashboard_users
        SET is_admin = $2,
            updated_at = NOW()
        WHERE email = $1
        RETURNING email, is_admin, created_at, updated_at, created_by
        ",
    )
    .bind(email)
    .bind(is_admin)
    .fetch_optional(pool)
    .await?)
}

pub async fn delete_dashboard_user(pool: &PgPool, email: &str) -> Result<bool, Error> {
    let result = sqlx::query("DELETE FROM tensorzero.dashboard_users WHERE email = $1")
        .bind(email)
        .execute(pool)
        .await?;
    Ok(result.rows_affected() > 0)
}

pub async fn count_dashboard_admins(pool: &PgPool) -> Result<i64, Error> {
    let count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*)::BIGINT FROM tensorzero.dashboard_users WHERE is_admin",
    )
    .fetch_one(pool)
    .await?;
    Ok(count)
}

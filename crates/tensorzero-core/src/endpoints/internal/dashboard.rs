// Modified by Delta-AI under Apache 2.0
//! Azure dashboard session and user-allowlist APIs.
//!
//! Gated by `TENSORZERO_UI_AZURE_AUTH`. The UI forwards the oauth2-proxy
//! `X-Auth-Request-Email` header so these handlers can identify the caller.

use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::config::gateway::{azure_auth_enabled, normalize_dashboard_email};
use crate::db::postgres::dashboard_users::{
    DashboardUserRow, count_dashboard_admins, delete_dashboard_user, get_dashboard_user,
    insert_dashboard_user, list_dashboard_users, seed_bootstrap_admins, set_dashboard_user_admin,
};
use crate::error::{Error, ErrorDetails};
use crate::utils::gateway::{AppState, AppStateData};

const EMAIL_HEADERS: &[&str] = &[
    "x-auth-request-email",
    "x-forwarded-email",
    "x-auth-request-preferred-username",
];

#[derive(Debug, Serialize)]
pub struct DashboardSessionResponse {
    pub enabled: bool,
    pub allowed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
    pub is_admin: bool,
}

#[derive(Debug, Serialize)]
pub struct ListDashboardUsersResponse {
    pub users: Vec<DashboardUserRow>,
}

#[derive(Debug, Deserialize)]
pub struct CreateDashboardUserRequest {
    pub email: String,
    #[serde(default)]
    pub is_admin: bool,
}

#[derive(Debug, Deserialize)]
pub struct UpdateDashboardUserRequest {
    pub email: String,
    pub is_admin: bool,
}

#[derive(Debug, Deserialize)]
pub struct DeleteDashboardUserRequest {
    pub email: String,
}

#[derive(Debug, Serialize)]
pub struct DeleteDashboardUserResponse {
    pub deleted: bool,
}

fn header_email(headers: &HeaderMap) -> Option<String> {
    for name in EMAIL_HEADERS {
        let Some(value) = headers.get(*name).and_then(|v| v.to_str().ok()) else {
            continue;
        };
        if let Some(email) = normalize_dashboard_email(value) {
            return Some(email);
        }
    }
    None
}

fn require_azure_auth() -> Result<(), Error> {
    if azure_auth_enabled() {
        Ok(())
    } else {
        Err(Error::new(ErrorDetails::InvalidRequest {
            message: "Azure dashboard auth is not enabled (`TENSORZERO_UI_AZURE_AUTH`)."
                .to_string(),
        }))
    }
}

fn require_email(headers: &HeaderMap) -> Result<String, Error> {
    header_email(headers).ok_or_else(|| {
        Error::new(ErrorDetails::InvalidRequest {
            message: "Missing Azure login email (`X-Auth-Request-Email`).".to_string(),
        })
    })
}

fn parse_email(raw: &str) -> Result<String, Error> {
    normalize_dashboard_email(raw).ok_or_else(|| {
        Error::new(ErrorDetails::InvalidRequest {
            message: format!("Invalid email `{raw}`"),
        })
    })
}

fn postgres_required(app_state: &AppStateData) -> Result<&sqlx::PgPool, Error> {
    app_state
        .postgres_connection_info
        .get_pool()
        .ok_or_else(|| {
            Error::new(ErrorDetails::PostgresConnection {
                message: "Postgres is required for dashboard user management".to_string(),
            })
        })
}

async fn seed_if_needed(app_state: &AppStateData) -> Result<(), Error> {
    let pool = postgres_required(app_state)?;
    seed_bootstrap_admins(pool, &app_state.config.gateway.ui.admin_emails).await
}

async fn require_admin(app_state: &AppStateData, headers: &HeaderMap) -> Result<String, Error> {
    require_azure_auth()?;
    seed_if_needed(app_state).await?;
    let email = require_email(headers)?;
    let pool = postgres_required(app_state)?;
    let Some(user) = get_dashboard_user(pool, &email).await? else {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: "Only dashboard admins can manage users".to_string(),
        }));
    };
    if !user.is_admin {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: "Only dashboard admins can manage users".to_string(),
        }));
    }
    Ok(email)
}

async fn ensure_not_last_admin(pool: &sqlx::PgPool, email: &str) -> Result<(), Error> {
    let Some(existing) = get_dashboard_user(pool, email).await? else {
        return Ok(());
    };
    if !existing.is_admin {
        return Ok(());
    }
    let admin_count = count_dashboard_admins(pool).await?;
    if admin_count <= 1 {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: "Cannot remove or demote the last dashboard admin".to_string(),
        }));
    }
    Ok(())
}

/// Handler for `GET /internal/dashboard/session`
#[instrument(name = "get_dashboard_session", skip_all)]
pub async fn get_dashboard_session_handler(
    State(app_state): AppState,
    headers: HeaderMap,
) -> Result<Json<DashboardSessionResponse>, Error> {
    if !azure_auth_enabled() {
        return Ok(Json(DashboardSessionResponse {
            enabled: false,
            allowed: true,
            email: None,
            is_admin: false,
        }));
    }

    seed_if_needed(&app_state).await?;
    let Some(email) = header_email(&headers) else {
        return Ok(Json(DashboardSessionResponse {
            enabled: true,
            allowed: false,
            email: None,
            is_admin: false,
        }));
    };

    let pool = postgres_required(&app_state)?;
    let user = get_dashboard_user(pool, &email).await?;
    let is_admin = user.as_ref().is_some_and(|user| user.is_admin);
    Ok(Json(DashboardSessionResponse {
        enabled: true,
        allowed: user.is_some(),
        email: Some(email),
        is_admin,
    }))
}

/// Handler for `GET /internal/dashboard/users`
#[instrument(name = "list_dashboard_users", skip_all)]
pub async fn list_dashboard_users_handler(
    State(app_state): AppState,
    headers: HeaderMap,
) -> Result<Json<ListDashboardUsersResponse>, Error> {
    require_admin(&app_state, &headers).await?;
    let pool = postgres_required(&app_state)?;
    let users = list_dashboard_users(pool).await?;
    Ok(Json(ListDashboardUsersResponse { users }))
}

/// Handler for `POST /internal/dashboard/users`
#[instrument(name = "create_dashboard_user", skip_all)]
pub async fn create_dashboard_user_handler(
    State(app_state): AppState,
    headers: HeaderMap,
    Json(payload): Json<CreateDashboardUserRequest>,
) -> Result<Json<DashboardUserRow>, Error> {
    let actor = require_admin(&app_state, &headers).await?;
    let email = parse_email(&payload.email)?;
    let pool = postgres_required(&app_state)?;
    let user = insert_dashboard_user(pool, &email, payload.is_admin, &actor).await?;
    Ok(Json(user))
}

/// Handler for `PATCH /internal/dashboard/users`
#[instrument(name = "update_dashboard_user", skip_all)]
pub async fn update_dashboard_user_handler(
    State(app_state): AppState,
    headers: HeaderMap,
    Json(payload): Json<UpdateDashboardUserRequest>,
) -> Result<Json<DashboardUserRow>, Error> {
    require_admin(&app_state, &headers).await?;
    let email = parse_email(&payload.email)?;
    let pool = postgres_required(&app_state)?;
    if !payload.is_admin {
        ensure_not_last_admin(pool, &email).await?;
    }
    let Some(user) = set_dashboard_user_admin(pool, &email, payload.is_admin).await? else {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!("Dashboard user `{email}` was not found"),
        }));
    };
    Ok(Json(user))
}

/// Handler for `POST /internal/dashboard/users/delete`
#[instrument(name = "delete_dashboard_user", skip_all)]
pub async fn delete_dashboard_user_handler(
    State(app_state): AppState,
    headers: HeaderMap,
    Json(payload): Json<DeleteDashboardUserRequest>,
) -> Result<Json<DeleteDashboardUserResponse>, Error> {
    require_admin(&app_state, &headers).await?;
    let email = parse_email(&payload.email)?;
    let pool = postgres_required(&app_state)?;
    ensure_not_last_admin(pool, &email).await?;
    if !delete_dashboard_user(pool, &email).await? {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!("Dashboard user `{email}` was not found"),
        }));
    }
    Ok(Json(DeleteDashboardUserResponse { deleted: true }))
}

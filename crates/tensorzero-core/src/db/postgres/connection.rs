// Modified by Delta-AI under Apache 2.0
//! Connection setup shared by the gateway and the migration command.
use std::sync::Arc;

use sqlx::postgres::{PgConnectOptions, PgPoolOptions};
use sqlx::{ConnectOptions, PgPool};
use tokio_util::task::AbortOnDropHandle;

#[cfg(feature = "aws-rds-iam")]
mod iam;

pub type TokenRefresh = Option<Arc<AbortOnDropHandle<()>>>;

pub(crate) fn configuration_error(message: &str) -> sqlx::Error {
    sqlx::Error::Configuration(message.into())
}

#[derive(Debug, PartialEq)]
enum AuthMode {
    Password,
    Iam,
}

impl AuthMode {
    fn parse(value: &str) -> Result<Self, sqlx::Error> {
        match value {
            "password" => Ok(Self::Password),
            "iam" => Ok(Self::Iam),
            _ => Err(configuration_error(
                "TENSORZERO_POSTGRES_AUTH_MODE must be `password` or `iam`",
            )),
        }
    }
}

/// Keep the returned refresh guard alive for as long as the pool is used.
/// SQLx clones share connect options, so rotation also applies to their new connections.
pub async fn connect(
    postgres_url: &str,
    pool_options: PgPoolOptions,
) -> Result<(PgPool, TokenRefresh), sqlx::Error> {
    let mode = AuthMode::parse(
        &std::env::var("TENSORZERO_POSTGRES_AUTH_MODE").unwrap_or_else(|_| "password".into()),
    )?;
    let options: PgConnectOptions = postgres_url.parse()?;
    let options = options.log_statements(tracing::log::LevelFilter::Trace);
    match mode {
        AuthMode::Password => Ok((pool_options.connect_with(options).await?, None)),
        AuthMode::Iam => {
            #[cfg(feature = "aws-rds-iam")]
            {
                iam::connect(options, pool_options).await
            }
            #[cfg(not(feature = "aws-rds-iam"))]
            Err(configuration_error(
                "IAM Postgres authentication requires a build with the `aws-rds-iam` feature",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use googletest::prelude::*;

    use super::AuthMode;

    #[gtest]
    fn authentication_mode_is_explicit() {
        expect_that!(
            AuthMode::parse("password").expect("valid mode"),
            eq(&AuthMode::Password)
        );
        expect_that!(
            AuthMode::parse("iam").expect("valid mode"),
            eq(&AuthMode::Iam)
        );
        expect_that!(AuthMode::parse("IAM").is_err(), eq(true));
        expect_that!(AuthMode::parse("").is_err(), eq(true));
    }
}

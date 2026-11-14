// Modified by Delta-AI under Apache 2.0
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use aws_credential_types::Credentials;
use aws_credential_types::provider::{ProvideCredentials, SharedCredentialsProvider};
use aws_sigv4::http_request::{
    SignableBody, SignableRequest, SignatureLocation, SigningSettings, sign,
};
use aws_sigv4::sign::v4;
use sqlx::PgPool;
use sqlx::postgres::{PgConnectOptions, PgPoolOptions, PgSslMode};
use tokio::time::{sleep, timeout};
use tokio_util::task::AbortOnDropHandle;
use url::Url;

use super::{TokenRefresh, configuration_error};

const REFRESH_INTERVAL: Duration = Duration::from_secs(5 * 60);
const RETRY_INTERVAL: Duration = Duration::from_secs(5);
const SIGN_TIMEOUT: Duration = Duration::from_secs(10);

#[async_trait]
trait TokenSource: Send + Sync {
    async fn token(&self) -> Result<String, sqlx::Error>;
}

struct RdsTokenSource {
    credentials: SharedCredentialsProvider,
    region: String,
    options: PgConnectOptions,
}

#[async_trait]
impl TokenSource for RdsTokenSource {
    async fn token(&self) -> Result<String, sqlx::Error> {
        let credentials = self.credentials.provide_credentials().await.map_err(|_| {
            configuration_error("Failed to load AWS credentials for RDS IAM authentication")
        })?;
        sign_token(&self.options, &self.region, credentials, SystemTime::now())
    }
}

fn configure(
    options: PgConnectOptions,
    region: &str,
    role: &str,
) -> Result<PgConnectOptions, sqlx::Error> {
    if region.trim().is_empty() {
        return Err(configuration_error(
            "AWS_REGION is required for RDS IAM authentication",
        ));
    }
    if !matches!(options.get_ssl_mode(), PgSslMode::VerifyFull) {
        return Err(configuration_error(
            "RDS IAM authentication requires sslmode=verify-full",
        ));
    }
    if options.get_host().starts_with('/') || options.get_username().is_empty() {
        return Err(configuration_error(
            "RDS IAM authentication requires a database hostname and user",
        ));
    }
    if role.is_empty()
        || role.len() > 63
        || !role
            .bytes()
            .all(|character| character.is_ascii_alphanumeric() || character == b'_')
        || role.as_bytes()[0].is_ascii_digit()
        || role == options.get_username()
    {
        return Err(configuration_error(
            "TENSORZERO_POSTGRES_ROLE must name the existing owner role, distinct from the IAM login (letters, digits, underscores; at most 63 bytes)",
        ));
    }
    // Startup options take effect before any query/migration. SQLx escapes option
    // values; role is also restricted above. The IAM login remains the signed user.
    Ok(options.options([("role", role)]))
}

fn sign_token(
    options: &PgConnectOptions,
    region: &str,
    credentials: Credentials,
    time: SystemTime,
) -> Result<String, sqlx::Error> {
    // Build query parameters through URL encoding; the resulting token is passed to
    // SQLx as a password, never interpolated into a Postgres URL.
    let mut url = Url::parse(&format!(
        "https://{}:{}/",
        options.get_host(),
        options.get_port()
    ))
    .map_err(|_| configuration_error("Invalid RDS IAM database endpoint"))?;
    url.query_pairs_mut()
        .append_pair("Action", "connect")
        .append_pair("DBUser", options.get_username());
    let identity = credentials.into();
    let mut settings = SigningSettings::default();
    settings.signature_location = SignatureLocation::QueryParams;
    settings.expires_in = Some(Duration::from_secs(15 * 60));
    let parameters = v4::SigningParams::builder()
        .identity(&identity)
        .region(region)
        .name("rds-db")
        .time(time)
        .settings(settings)
        .build()
        .map_err(|_| configuration_error("Failed to configure RDS IAM signing"))?;
    let request = SignableRequest::new(
        "GET",
        url.as_str(),
        std::iter::empty(),
        SignableBody::Bytes(&[]),
    )
    .map_err(|_| configuration_error("Failed to construct RDS IAM signing request"))?;
    let (instructions, _) = sign(request, &parameters.into())
        .map_err(|_| configuration_error("Failed to sign RDS IAM token"))?
        .into_parts();
    for (name, value) in instructions.params() {
        url.query_pairs_mut().append_pair(name, value);
    }
    Ok(url.as_str().trim_start_matches("https://").to_owned())
}

async fn next_token(source: &dyn TokenSource) -> Result<String, sqlx::Error> {
    timeout(SIGN_TIMEOUT, source.token())
        .await
        .map_err(|_| configuration_error("Timed out obtaining an RDS IAM token"))?
}

/// SQLx 0.9 has no before-connect hook. Replace its shared connection options
/// well before the 15-minute token expiry. Existing sessions remain valid.
#[expect(
    clippy::disallowed_methods,
    reason = "refresh has no writes to drain; its guard aborts it and pool close stops it"
)]
fn spawn_refresh(pool: PgPool, source: Arc<dyn TokenSource>) -> Arc<AbortOnDropHandle<()>> {
    Arc::new(AbortOnDropHandle::new(tokio::spawn(async move {
        let mut delay = REFRESH_INTERVAL;
        loop {
            tokio::select! {
                biased;
                () = pool.close_event() => return,
                () = sleep(delay) => {},
            }
            let result = tokio::select! {
                biased;
                () = pool.close_event() => return,
                result = next_token(source.as_ref()) => result,
            };
            match result {
                Ok(token) => {
                    let options = pool.connect_options().as_ref().clone().password(&token);
                    pool.set_connect_options(options);
                    delay = REFRESH_INTERVAL;
                }
                Err(_) => {
                    // Never log credentials/token URLs or fall back to a static password.
                    // Existing sessions continue; after expiry RDS rejects new connections
                    // until credentials recover and a later retry replaces the token.
                    tracing::warn!("RDS IAM token refresh failed; retrying in 5 seconds");
                    delay = RETRY_INTERVAL;
                }
            }
        }
    })))
}

pub(super) async fn connect(
    options: PgConnectOptions,
    pool_options: PgPoolOptions,
) -> Result<(PgPool, TokenRefresh), sqlx::Error> {
    let region = std::env::var("AWS_REGION").unwrap_or_default();
    let role = std::env::var("TENSORZERO_POSTGRES_ROLE").unwrap_or_default();
    let options = configure(options, &region, &role)?;
    let config = aws_config::defaults(aws_config::BehaviorVersion::latest())
        .region(aws_config::Region::new(region.clone()))
        .load()
        .await;
    let credentials = config.credentials_provider().ok_or_else(|| {
        configuration_error("No AWS credentials provider for RDS IAM authentication")
    })?;
    let source = Arc::new(RdsTokenSource {
        credentials,
        region,
        options: options.clone(),
    });
    let token = next_token(source.as_ref()).await?;
    let pool = pool_options.connect_lazy_with(options.password(&token));
    let refresh = spawn_refresh(pool.clone(), source);
    // Validate connectivity before returning. Dropping the guard on error aborts
    // the task, which releases its pool clone; there is no Arc ownership cycle.
    let _connection = pool.acquire().await?;
    Ok((pool, Some(refresh)))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::result::Result;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use googletest::prelude::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::mpsc;
    use tokio::time::{advance, pause, resume};

    use super::*;

    #[gtest]
    fn validates_tls_region_and_owner_without_loading_credentials() {
        let options = PgConnectOptions::new()
            .host("example.rds.amazonaws.com")
            .username("iam_user");
        expect_that!(
            configure(options.clone(), "us-west-2", "existing_owner").is_err(),
            eq(true)
        );
        let options = options.ssl_mode(PgSslMode::VerifyFull);
        expect_that!(
            configure(options.clone(), "", "existing_owner").is_err(),
            eq(true)
        );
        expect_that!(
            configure(options.clone(), "us-west-2", "existing_owner").is_ok(),
            eq(true)
        );
        for role in ["", "iam_user", "role;DROP", "owner with space", "1owner"] {
            expect_that!(
                configure(options.clone(), "us-west-2", role).is_err(),
                eq(true)
            );
        }
    }

    #[gtest]
    fn signs_exact_endpoint_user_and_session_credentials() {
        let options = PgConnectOptions::new()
            .host("example.rds.amazonaws.com")
            .port(5433)
            .username("iam/user+name");
        let credentials = Credentials::new(
            "test-access",
            "test-secret",
            Some("session+/=".into()),
            None,
            "test",
        );
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        let token =
            sign_token(&options, "us-west-2", credentials.clone(), now).expect("sign token");
        let url = Url::parse(&format!("https://{token}")).expect("token is a URL without scheme");
        let query: HashMap<_, _> = url.query_pairs().into_owned().collect();
        expect_that!(url.host_str(), some(eq("example.rds.amazonaws.com")));
        expect_that!(url.port(), some(eq(5433)));
        expect_that!(query["DBUser"].as_str(), eq("iam/user+name"));
        expect_that!(query["Action"].as_str(), eq("connect"));
        expect_that!(query["X-Amz-Expires"].as_str(), eq("900"));
        expect_that!(query["X-Amz-Security-Token"].as_str(), eq("session+/="));
        expect_that!(
            query["X-Amz-Credential"].as_str(),
            eq("test-access/20231114/us-west-2/rds-db/aws4_request")
        );
        expect_that!(query["X-Amz-SignedHeaders"].as_str(), eq("host"));
        let later = sign_token(
            &options,
            "us-west-2",
            credentials,
            now + Duration::from_secs(960),
        )
        .expect("sign again after the first token expires");
        expect_that!(later, not(eq(&token)));
    }

    struct RecoveringSource {
        calls: AtomicUsize,
        events: Option<mpsc::UnboundedSender<usize>>,
    }

    #[async_trait]
    impl TokenSource for RecoveringSource {
        async fn token(&self) -> Result<String, sqlx::Error> {
            let attempt = self.calls.fetch_add(1, Ordering::SeqCst);
            if let Some(events) = &self.events {
                let _ = events.send(attempt + 1);
            }
            if attempt == 0 {
                return Err(configuration_error("temporary credential failure"));
            }
            Ok(format!("refreshed-token-{attempt}+/=?&"))
        }
    }

    // A tiny Postgres authentication server proves which password is actually
    // sent on two separate physical connections, rather than only inspecting options.
    #[expect(
        clippy::disallowed_methods,
        reason = "test server is aborted by its guard"
    )]
    async fn authentication_server() -> (u16, mpsc::Receiver<String>, AbortOnDropHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test listener");
        let port = listener.local_addr().expect("listener address").port();
        let (sender, receiver) = mpsc::channel(4);
        let task = tokio::spawn(async move {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().await.expect("accept connection");
                let length = stream.read_u32().await.expect("startup length");
                let mut startup = vec![0; (length - 4) as usize];
                stream
                    .read_exact(&mut startup)
                    .await
                    .expect("startup packet");
                for parameter in [
                    b"user\0iam_user\0".as_slice(),
                    b"options\0-c role=existing_owner\0".as_slice(),
                ] {
                    assert_that!(
                        startup
                            .windows(parameter.len())
                            .any(|bytes| bytes == parameter),
                        eq(true),
                        "each physical connection must authenticate as the IAM login and assume the existing owner before SQL"
                    );
                }
                stream
                    .write_all(b"R\0\0\0\x08\0\0\0\x03")
                    .await
                    .expect("request password");
                assert_that!(stream.read_u8().await.expect("password type"), eq(b'p'));
                let length = stream.read_u32().await.expect("password length");
                let mut password = vec![0; (length - 4) as usize];
                stream
                    .read_exact(&mut password)
                    .await
                    .expect("password packet");
                password.pop();
                sender
                    .send(String::from_utf8(password).expect("UTF-8 password"))
                    .await
                    .expect("record password");
                stream
                    .write_all(b"R\0\0\0\x08\0\0\0\0Z\0\0\0\x05I")
                    .await
                    .expect("ready");
                let mut remaining = Vec::new();
                stream
                    .read_to_end(&mut remaining)
                    .await
                    .expect("wait for client close");
            }
        });
        (port, receiver, AbortOnDropHandle::new(task))
    }

    #[gtest]
    #[tokio::test]
    async fn refreshed_token_is_used_by_new_connections_from_a_pool_clone() {
        let (port, mut passwords, _server) = authentication_server().await;
        let options = PgConnectOptions::new()
            .host("127.0.0.1")
            .port(port)
            .ssl_mode(PgSslMode::VerifyFull)
            .username("iam_user")
            .password("initial-token");
        let options = configure(options, "us-west-2", "existing_owner")
            .expect("configure IAM owner role")
            .ssl_mode(PgSslMode::Disable);
        let pool = PgPoolOptions::new()
            .max_connections(1)
            .test_before_acquire(false)
            .idle_timeout(None)
            .max_lifetime(None)
            .connect_lazy_with(options);
        let first = pool.acquire().await.expect("first physical connection");
        expect_that!(
            passwords.recv().await.expect("first password"),
            eq("initial-token")
        );
        first
            .close()
            .await
            .expect("close first physical connection");
        let (events, mut refreshed) = mpsc::unbounded_channel();
        let source = Arc::new(RecoveringSource {
            calls: AtomicUsize::new(0),
            events: Some(events),
        });
        let weak_source = Arc::downgrade(&source);
        let refresh = spawn_refresh(pool.clone(), source.clone());
        let cloned_pool = pool.clone();
        pause();
        tokio::task::yield_now().await;
        advance(REFRESH_INTERVAL).await;
        expect_that!(
            refreshed.recv().await.expect("failed refresh observed"),
            eq(1)
        );
        advance(RETRY_INTERVAL).await;
        expect_that!(
            refreshed.recv().await.expect("recovered refresh observed"),
            eq(2)
        );
        // Cross the original token's 15-minute lifetime without restarting the pool.
        for expected in 3..=4 {
            advance(REFRESH_INTERVAL).await;
            expect_that!(
                refreshed.recv().await.expect("periodic refresh observed"),
                eq(expected)
            );
        }
        advance(Duration::from_secs(55)).await;
        tokio::task::yield_now().await;
        expect_that!(source.calls.load(Ordering::SeqCst), eq(4));
        resume();
        let second = cloned_pool
            .acquire()
            .await
            .expect("second physical connection after refresh recovery");
        expect_that!(
            passwords.recv().await.expect("second password"),
            eq("refreshed-token-3+/=?&")
        );
        second.close().await.expect("close second connection");
        drop(source);
        pool.close().await;
        tokio::task::yield_now().await;
        expect_that!(
            weak_source.upgrade().is_none(),
            eq(true),
            "pool close terminates refresh even with its guard alive"
        );
        drop(refresh);
    }

    #[gtest]
    #[tokio::test]
    async fn dropping_the_last_guard_releases_the_refresh_task() {
        let pool = PgPoolOptions::new().connect_lazy_with(PgConnectOptions::new());
        let source = Arc::new(RecoveringSource {
            calls: AtomicUsize::new(0),
            events: None,
        });
        let weak_source = Arc::downgrade(&source);
        let refresh = spawn_refresh(pool.clone(), source);
        let second_guard = refresh.clone();
        drop(refresh);
        expect_that!(weak_source.upgrade().is_some(), eq(true));
        drop(second_guard);
        tokio::task::yield_now().await;
        expect_that!(
            weak_source.upgrade().is_none(),
            eq(true),
            "no task/pool ownership cycle remains"
        );
        pool.close().await;
    }
}

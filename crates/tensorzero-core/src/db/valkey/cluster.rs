// Modified by Delta-AI under Apache 2.0
//! Bounded cluster commands with recovery through the configured seed endpoint.
//!
//! A managed Redis failover can retire every advertised shard address at once.
//! Retrying those addresses cannot discover the new topology. All clones share
//! one replaceable connection and one recovery loop. Failed commands are NEVER
//! replayed here: a timed-out XADD may already have committed on the server.

use std::sync::Arc;
use std::time::Duration;

use redis::aio::ConnectionLike;
use redis::cluster::ClusterClient;
use redis::cluster_async::ClusterConnection;
use redis::{Cmd, Pipeline, RedisError, RedisResult, Value};
use tokio::sync::RwLock;
use tokio::time::{sleep, timeout};

use super::ASYNC_INFERENCE_STREAM_RESPONSE_TIMEOUT;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_BACKOFF: Duration = Duration::from_secs(30);

#[derive(Clone)]
pub struct RecoveringClusterConnection {
    shared: Arc<Shared>,
}

struct Shared {
    client: ClusterClient,
    state: RwLock<State>,
    command_timeout: Duration,
    connect_timeout: Duration,
}

struct State {
    connection: ClusterConnection,
    generation: u64,
    recovering: bool,
}

fn deadline_error() -> RedisError {
    std::io::Error::new(
        std::io::ErrorKind::TimedOut,
        "Valkey cluster operation timed out",
    )
    .into()
}

impl RecoveringClusterConnection {
    pub async fn new(url: String) -> RedisResult<Self> {
        Self::with_timeouts(
            url,
            ASYNC_INFERENCE_STREAM_RESPONSE_TIMEOUT,
            CONNECT_TIMEOUT,
        )
        .await
    }

    async fn with_timeouts(
        url: String,
        command_timeout: Duration,
        connect_timeout: Duration,
    ) -> RedisResult<Self> {
        let client = ClusterClient::builder(vec![url])
            // Azure advertises IPs while issuing certificates for the DNS name.
            // Preserve certificate-chain verification and existing TLS policy.
            .danger_accept_invalid_hostnames(true)
            .connection_timeout(connect_timeout)
            .response_timeout(command_timeout)
            .overall_response_timeout(Some(command_timeout))
            // Do not replay writes with an ambiguous outcome. A future command
            // uses the refreshed topology after an error instead.
            .retries(0)
            .build()?;
        let connection = timeout(connect_timeout, client.get_async_connection())
            .await
            .map_err(|_| deadline_error())??;
        Ok(Self {
            shared: Arc::new(Shared {
                client,
                state: RwLock::new(State {
                    connection,
                    generation: 0,
                    recovering: false,
                }),
                command_timeout,
                connect_timeout,
            }),
        })
    }

    async fn snapshot(&self) -> RedisResult<(ClusterConnection, u64)> {
        let state = self.shared.state.read().await;
        if state.recovering {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotConnected,
                "Valkey cluster connection is recovering",
            )
            .into());
        }
        Ok((state.connection.clone(), state.generation))
    }

    async fn recover_after_error(&self, generation: u64, error: &RedisError) {
        // Redis application errors (e.g. WRONGTYPE) cannot be fixed by reconnecting.
        if !(error.is_io_error()
            || error.is_timeout()
            || error.kind() == redis::ErrorKind::ClusterConnectionNotFound
            || matches!(
                error.code(),
                Some("MOVED" | "ASK" | "CLUSTERDOWN" | "MASTERDOWN")
            ))
        {
            return;
        }
        let mut state = self.shared.state.write().await;
        if state.generation != generation || state.recovering {
            return;
        }
        state.recovering = true;
        let weak = Arc::downgrade(&self.shared);
        tracing::warn!(
            generation,
            "Rebuilding Valkey cluster connection from seed endpoint"
        );
        #[expect(
            clippy::disallowed_methods,
            reason = "Recovery only owns a weak connection reference, has no business writes, and may be cancelled on shutdown"
        )]
        tokio::spawn(async move {
            let mut backoff = Duration::from_secs(1);
            loop {
                let Some(shared) = weak.upgrade() else {
                    return;
                };
                let result =
                    timeout(shared.connect_timeout, shared.client.get_async_connection()).await;
                if let Ok(Ok(connection)) = result {
                    let mut state = shared.state.write().await;
                    state.connection = connection;
                    state.generation += 1;
                    state.recovering = false;
                    tracing::info!(
                        generation = state.generation,
                        "Recovered Valkey cluster connection from seed endpoint"
                    );
                    return;
                }
                // No URL or credentials in diagnostics. The weak reference
                // lets the loop exit when the gateway/config is dropped.
                tracing::warn!(
                    retry_after_secs = backoff.as_secs(),
                    "Valkey cluster seed reconnect failed"
                );
                drop(shared);
                sleep(backoff).await;
                backoff = (backoff * 2).min(MAX_BACKOFF);
            }
        });
    }

    pub(super) async fn command(&self, cmd: &Cmd, deadline: Duration) -> RedisResult<Value> {
        let (mut connection, generation) = self.snapshot().await?;
        let result = timeout(deadline, connection.req_packed_command(cmd))
            .await
            .unwrap_or_else(|_| Err(deadline_error()));
        if let Err(error) = &result {
            self.recover_after_error(generation, error).await;
        }
        result
    }
}

impl ConnectionLike for RecoveringClusterConnection {
    fn req_packed_command<'a>(&'a mut self, cmd: &'a Cmd) -> redis::RedisFuture<'a, Value> {
        Box::pin(async move { self.command(cmd, self.shared.command_timeout).await })
    }

    fn req_packed_commands<'a>(
        &'a mut self,
        cmd: &'a Pipeline,
        offset: usize,
        count: usize,
    ) -> redis::RedisFuture<'a, Vec<Value>> {
        Box::pin(async move {
            let (mut connection, generation) = self.snapshot().await?;
            let result = timeout(
                self.shared.command_timeout,
                connection.req_packed_commands(cmd, offset, count),
            )
            .await
            .unwrap_or_else(|_| Err(deadline_error()));
            if let Err(error) = &result {
                self.recover_after_error(generation, error).await;
            }
            result
        })
    }

    fn get_db(&self) -> i64 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use googletest::prelude::*;
    use redis::AsyncCommands;
    use std::sync::atomic::{AtomicBool, AtomicU16, AtomicUsize, Ordering};
    use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
    use tokio::net::{TcpListener, TcpStream};
    use tokio::task::{JoinHandle, JoinSet};

    struct Node {
        port: u16,
        blackhole: Arc<AtomicBool>,
        task: JoinHandle<()>,
    }

    impl Drop for Node {
        fn drop(&mut self) {
            self.task.abort();
        }
    }

    async fn command(
        reader: &mut BufReader<tokio::net::tcp::OwnedReadHalf>,
    ) -> Option<Vec<String>> {
        let mut line = String::new();
        reader.read_line(&mut line).await.ok()?;
        let count: usize = line.strip_prefix('*')?.trim().parse().ok()?;
        let mut args = Vec::new();
        for _ in 0..count {
            line.clear();
            reader.read_line(&mut line).await.ok()?;
            let size: usize = line.strip_prefix('$')?.trim().parse().ok()?;
            let mut data = vec![0; size + 2];
            reader.read_exact(&mut data).await.ok()?;
            args.push(String::from_utf8(data[..size].to_vec()).ok()?);
        }
        Some(args)
    }

    async fn serve(
        stream: TcpStream,
        topology: Arc<AtomicU16>,
        blackhole: Arc<AtomicBool>,
        discoveries: Arc<AtomicUsize>,
        writes: Arc<AtomicUsize>,
    ) {
        let (reader, mut writer) = stream.into_split();
        let mut reader = BufReader::new(reader);
        while let Some(args) = command(&mut reader).await {
            let name = args[0].to_ascii_uppercase();
            if name == "INCR" || name == "INCRBY" {
                writes.fetch_add(1, Ordering::SeqCst);
            }
            if blackhole.load(Ordering::SeqCst) {
                std::future::pending::<()>().await;
            }
            let response = match name.as_str() {
                "CLUSTER" => {
                    discoveries.fetch_add(1, Ordering::SeqCst);
                    format!(
                        "*1\r\n*3\r\n:0\r\n:16383\r\n*2\r\n$9\r\n127.0.0.1\r\n:{}\r\n",
                        topology.load(Ordering::SeqCst)
                    )
                }
                "PING" => "+PONG\r\n".to_string(),
                "GET" => "-WRONGTYPE test application error\r\n".to_string(),
                "EXISTS" | "INCR" | "INCRBY" => ":1\r\n".to_string(),
                _ => "+OK\r\n".to_string(),
            };
            if writer.write_all(response.as_bytes()).await.is_err() {
                return;
            }
        }
    }

    #[expect(
        clippy::disallowed_methods,
        reason = "Fake Redis listener is aborted by the test fixture Drop"
    )]
    async fn node(
        topology: Arc<AtomicU16>,
        discoveries: Arc<AtomicUsize>,
        writes: Arc<AtomicUsize>,
    ) -> Node {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind fake Redis node");
        let port = listener.local_addr().expect("bound address").port();
        let blackhole = Arc::new(AtomicBool::new(false));
        let gate = blackhole.clone();
        let task = tokio::spawn(async move {
            let mut clients = JoinSet::new();
            loop {
                tokio::select! {
                    accept = listener.accept() => {
                        let Ok((stream, _)) = accept else { break; };
                        clients.spawn(serve(stream, topology.clone(), gate.clone(), discoveries.clone(), writes.clone()));
                    }
                    _ = clients.join_next(), if !clients.is_empty() => {}
                }
            }
        });
        Node {
            port,
            blackhole,
            task,
        }
    }

    #[gtest]
    #[tokio::test]
    async fn recovers_retired_shard_from_seed_without_replaying_ambiguous_write() {
        let topology = Arc::new(AtomicU16::new(0));
        let discoveries = Arc::new(AtomicUsize::new(0));
        let writes = Arc::new(AtomicUsize::new(0));
        let old = node(topology.clone(), discoveries.clone(), writes.clone()).await;
        let new = node(topology.clone(), discoveries.clone(), writes.clone()).await;
        let seed = node(topology.clone(), discoveries.clone(), writes.clone()).await;
        topology.store(old.port, Ordering::SeqCst);
        let mut conn = RecoveringClusterConnection::with_timeouts(
            format!("redis://127.0.0.1:{}", seed.port),
            Duration::from_millis(200),
            Duration::from_secs(2),
        )
        .await
        .expect("initial topology discovery");
        let mut existing_clone = conn.clone();
        let before: bool = conn.exists("test-key").await.expect("old shard responds");
        expect_that!(before, eq(true));

        // No MOVED is delivered: the old shard accepts a write but never
        // replies. Only the original seed knows the replacement address.
        topology.store(new.port, Ordering::SeqCst);
        old.blackhole.store(true, Ordering::SeqCst);
        let result: RedisResult<i64> = timeout(Duration::from_secs(1), conn.incr("counter", 1))
            .await
            .expect("command must have an overall deadline");
        expect_that!(result.is_err(), eq(true));
        timeout(Duration::from_secs(5), async {
            loop {
                let result: RedisResult<bool> = existing_clone.exists("test-key").await;
                if matches!(result, Ok(true)) {
                    break;
                }
                sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("existing clone recovers without restarting gateway");
        expect_that!(
            writes.load(Ordering::SeqCst),
            eq(1),
            "timed-out write must not be replayed"
        );
        expect_that!(
            conn.shared.state.read().await.generation,
            eq(1),
            "one shared rebuild"
        );
        expect_that!(discoveries.load(Ordering::SeqCst), ge(2));
    }

    #[gtest]
    #[tokio::test]
    async fn unavailable_seed_recovers_in_background_with_one_shared_rebuild() {
        let topology = Arc::new(AtomicU16::new(0));
        let discoveries = Arc::new(AtomicUsize::new(0));
        let writes = Arc::new(AtomicUsize::new(0));
        let old = node(topology.clone(), discoveries.clone(), writes.clone()).await;
        let new = node(topology.clone(), discoveries.clone(), writes.clone()).await;
        let seed = node(topology.clone(), discoveries.clone(), writes).await;
        topology.store(old.port, Ordering::SeqCst);
        let conn = RecoveringClusterConnection::with_timeouts(
            format!("redis://127.0.0.1:{}", seed.port),
            Duration::from_millis(100),
            Duration::from_millis(200),
        )
        .await
        .expect("initial connection");
        topology.store(new.port, Ordering::SeqCst);
        old.blackhole.store(true, Ordering::SeqCst);
        seed.blackhole.store(true, Ordering::SeqCst);
        let results = futures::future::join_all((0..8).map(|_| {
            let mut clone = conn.clone();
            async move { clone.exists::<_, bool>("key").await }
        }))
        .await;
        expect_that!(results.iter().all(|result| result.is_err()), eq(true));
        sleep(Duration::from_millis(300)).await;
        expect_that!(conn.shared.state.read().await.recovering, eq(true));
        seed.blackhole.store(false, Ordering::SeqCst);
        // No further command triggers recovery: the background loop must
        // revisit the seed on its own after its first attempt timed out.
        timeout(Duration::from_secs(5), async {
            while conn.shared.state.read().await.recovering {
                sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("seed recovery without any caller retry");
        expect_that!(conn.shared.state.read().await.generation, eq(1));
        let mut clone = conn.clone();
        expect_that!(
            clone
                .exists::<_, bool>("key")
                .await
                .expect("replacement shard"),
            eq(true)
        );
    }

    #[gtest]
    #[tokio::test]
    async fn application_errors_do_not_rebuild_healthy_connections() {
        let topology = Arc::new(AtomicU16::new(0));
        let discoveries = Arc::new(AtomicUsize::new(0));
        let writes = Arc::new(AtomicUsize::new(0));
        let seed = node(topology.clone(), discoveries.clone(), writes).await;
        topology.store(seed.port, Ordering::SeqCst);
        let mut conn = RecoveringClusterConnection::new(format!("redis://127.0.0.1:{}", seed.port))
            .await
            .expect("initial connection");
        let result: RedisResult<String> = conn.get("wrong-type").await;
        expect_that!(
            result.expect_err("server application error").code(),
            some(eq("WRONGTYPE"))
        );
        expect_that!(conn.shared.state.read().await.generation, eq(0));
        expect_that!(conn.shared.state.read().await.recovering, eq(false));
    }
}

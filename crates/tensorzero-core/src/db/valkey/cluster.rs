// Modified by Delta-AI under Apache 2.0
//! Bounded cluster commands with recovery through the configured seed endpoint.
//!
//! A managed Redis failover can retire every advertised shard address at once.
//! Retrying those addresses cannot discover the new topology. All clones share
//! one replaceable connection and one recovery loop. Ambiguous failures are NEVER
//! replayed: a timed-out XADD may already have committed. Explicit ASK/MOVED
//! rejections are followed within the original deadline and a redirect budget.

use std::sync::Arc;
use std::time::Duration;

use redis::aio::ConnectionLike;
use redis::cluster::ClusterClient;
use redis::cluster_async::ClusterConnection;
use redis::{
    AsyncConnectionConfig, Client, Cmd, ConnectionAddr, ConnectionInfo, IntoConnectionInfo,
    Pipeline, RedisError, RedisResult, Value,
};
use tokio::sync::RwLock;
use tokio::time::{sleep, timeout};

use super::ASYNC_INFERENCE_STREAM_RESPONSE_TIMEOUT;

const MAX_REDIRECTS: usize = 3;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
// Bootstrap connects the seed, discovers slots, and checks advertised peers.
// Give redis-rs time to discard unavailable replicas after their own connection
// and response deadlines. Reusing CONNECT_TIMEOUT would cancel discovery first.
const CLUSTER_BOOTSTRAP_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_BACKOFF: Duration = Duration::from_secs(30);

#[derive(Clone)]
pub struct RecoveringClusterConnection {
    shared: Arc<Shared>,
}

struct Shared {
    client: ClusterClient,
    seed: ConnectionInfo,
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
        let seed = url.as_str().into_connection_info()?;
        let mut address = seed.addr().clone();
        address.set_danger_accept_invalid_hostnames(true);
        let seed = seed.set_addr(address);
        let client = ClusterClient::builder(vec![url])
            // Azure advertises IPs while issuing certificates for the DNS name.
            // Preserve certificate-chain verification and existing TLS policy.
            .danger_accept_invalid_hostnames(true)
            .connection_timeout(connect_timeout)
            .response_timeout(command_timeout)
            .overall_response_timeout(Some(command_timeout))
            // Disable generic retries, which also replay ambiguous writes.
            // command_with_redirects follows explicit rejections separately.
            .retries(0)
            .build()?;
        let connection = timeout(CLUSTER_BOOTSTRAP_TIMEOUT, client.get_async_connection())
            .await
            .map_err(|_| deadline_error())??;
        Ok(Self {
            shared: Arc::new(Shared {
                client,
                seed,
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
                let result = timeout(
                    CLUSTER_BOOTSTRAP_TIMEOUT,
                    shared.client.get_async_connection(),
                )
                .await;
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
        let result = timeout(
            deadline,
            command_with_redirects(
                &mut connection,
                cmd,
                &self.shared.seed,
                self.shared.connect_timeout,
                deadline,
            ),
        )
        .await
        .unwrap_or_else(|_| Err(deadline_error()));
        if let Err(error) = &result {
            self.recover_after_error(generation, error).await;
        }
        result
    }
}

/// Only follow explicit redirects: Redis has rejected these commands without
/// executing them. IO errors and timeouts never enter this retry loop.
async fn command_with_redirects(
    connection: &mut ClusterConnection,
    cmd: &Cmd,
    seed: &ConnectionInfo,
    connect_timeout: Duration,
    command_timeout: Duration,
) -> RedisResult<Value> {
    let mut result = connection.req_packed_command(cmd).await;
    // Limit manual redirects to the single-node commands used by async
    // streams. A multi-node command may already have written on other shards,
    // so its aggregate error does not authorize replaying the whole command.
    let mut args = cmd.args_iter();
    let safe_to_redirect = match args.next() {
        Some(redis::Arg::Simple(name)) => match name.to_ascii_uppercase().as_slice() {
            b"DEL" | b"EXISTS" => args.len() == 1,
            b"XADD" | b"XRANGE" | b"XREAD" | b"EXPIRE" | b"GET" | b"INCR" | b"INCRBY" => true,
            _ => false,
        },
        _ => false,
    };
    if !safe_to_redirect {
        return result;
    }
    for _ in 0..MAX_REDIRECTS {
        let Err(error) = &result else {
            return result;
        };
        let asking = match error.code() {
            Some("ASK") => true,
            Some("MOVED") => false,
            _ => return result,
        };
        let Some((address, _slot)) = error.redirect_node() else {
            return result;
        };
        let Some((host, port)) = address.rsplit_once(':') else {
            return result;
        };
        let Ok(port) = port.parse::<u16>() else {
            return result;
        };
        // Open a connection to the actual redirect target, which may not be
        // in CLUSTER SLOTS yet. Copy authentication, protocol and TLS settings
        // from the configured seed; never use a reconnecting manager here.
        let mut address = seed.addr().clone();
        match &mut address {
            ConnectionAddr::Tcp(target_host, target_port)
            | ConnectionAddr::TcpTls {
                host: target_host,
                port: target_port,
                ..
            } => {
                host.trim_start_matches('[')
                    .trim_end_matches(']')
                    .clone_into(target_host);
                *target_port = port;
            }
            _ => return result,
        }
        let mut redirected = Client::open(seed.clone().set_addr(address))?
            .get_multiplexed_async_connection_with_config(
                &AsyncConnectionConfig::new()
                    .set_connection_timeout(Some(connect_timeout))
                    .set_response_timeout(Some(command_timeout)),
            )
            .await?;
        if asking {
            // This connection is exclusive to this redirect, so no other
            // caller can insert a command between ASKING and the data command.
            redis::cmd("ASKING")
                .query_async::<()>(&mut redirected)
                .await?;
        }
        result = redirected
            .req_packed_command(cmd)
            .await
            .and_then(Value::extract_error);
    }
    result
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

    #[derive(Default)]
    struct NodeBehavior {
        replica: AtomicU16,
        stall_readonly: AtomicBool,
        readonly_requests: AtomicUsize,
        target: AtomicU16,
        moved: AtomicBool,
        require_asking: AtomicBool,
        asking_count: AtomicUsize,
        requests: AtomicUsize,
        stall_writes: AtomicBool,
    }

    struct Node {
        port: u16,
        blackhole: Arc<AtomicBool>,
        task: JoinHandle<()>,
        redirect: Arc<NodeBehavior>,
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
        redirect: Arc<NodeBehavior>,
    ) {
        let (reader, mut writer) = stream.into_split();
        let mut reader = BufReader::new(reader);
        let mut asked = false;
        while let Some(args) = command(&mut reader).await {
            let name = args[0].to_ascii_uppercase();
            if name == "READONLY" && redirect.stall_readonly.load(Ordering::SeqCst) {
                redirect.readonly_requests.fetch_add(1, Ordering::SeqCst);
                std::future::pending::<()>().await;
            }
            let data_command = matches!(name.as_str(), "EXISTS" | "INCR" | "INCRBY");
            if data_command {
                redirect.requests.fetch_add(1, Ordering::SeqCst);
                let target = redirect.target.load(Ordering::SeqCst);
                if target != 0 {
                    let kind = if redirect.moved.load(Ordering::SeqCst) {
                        "MOVED"
                    } else {
                        "ASK"
                    };
                    if writer
                        .write_all(format!("-{kind} 123 127.0.0.1:{target}\r\n").as_bytes())
                        .await
                        .is_err()
                    {
                        return;
                    }
                    continue;
                }
                if redirect.require_asking.load(Ordering::SeqCst) && !asked {
                    if writer.write_all(b"-ERR ASKING required\r\n").await.is_err() {
                        return;
                    }
                    continue;
                }
                asked = false;
            }
            if name == "INCR" || name == "INCRBY" {
                writes.fetch_add(1, Ordering::SeqCst);
                if redirect.stall_writes.load(Ordering::SeqCst) {
                    std::future::pending::<()>().await;
                }
            }
            if blackhole.load(Ordering::SeqCst) {
                std::future::pending::<()>().await;
            }
            let response = match name.as_str() {
                "CLUSTER" => {
                    discoveries.fetch_add(1, Ordering::SeqCst);
                    let replica = redirect.replica.load(Ordering::SeqCst);
                    let node_count = if replica == 0 { 3 } else { 4 };
                    let mut slots = format!(
                        "*1\r\n*{node_count}\r\n:0\r\n:16383\r\n*2\r\n$9\r\n127.0.0.1\r\n:{}\r\n",
                        topology.load(Ordering::SeqCst)
                    );
                    if replica != 0 {
                        slots.push_str(&format!("*2\r\n$9\r\n127.0.0.1\r\n:{replica}\r\n"));
                    }
                    slots
                }
                "ASKING" => {
                    asked = true;
                    redirect.asking_count.fetch_add(1, Ordering::SeqCst);
                    "+OK\r\n".to_string()
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
        let redirect = Arc::new(NodeBehavior::default());
        let behavior = redirect.clone();
        let task = tokio::spawn(async move {
            let mut clients = JoinSet::new();
            loop {
                tokio::select! {
                    accept = listener.accept() => {
                        let Ok((stream, _)) = accept else { break; };
                        clients.spawn(serve(stream, topology.clone(), gate.clone(), discoveries.clone(), writes.clone(), behavior.clone()));
                    }
                    _ = clients.join_next(), if !clients.is_empty() => {}
                }
            }
        });
        Node {
            port,
            blackhole,
            task,
            redirect,
        }
    }

    #[gtest]
    #[tokio::test]
    async fn starts_with_unresponsive_replica_and_healthy_primary() {
        let topology = Arc::new(AtomicU16::new(0));
        let discoveries = Arc::new(AtomicUsize::new(0));
        let writes = Arc::new(AtomicUsize::new(0));
        let primary = node(topology.clone(), discoveries.clone(), writes.clone()).await;
        let replica = node(topology.clone(), discoveries.clone(), writes.clone()).await;
        topology.store(primary.port, Ordering::SeqCst);
        primary
            .redirect
            .replica
            .store(replica.port, Ordering::SeqCst);
        replica
            .redirect
            .stall_readonly
            .store(true, Ordering::SeqCst);

        let mut connection = RecoveringClusterConnection::with_timeouts(
            format!("redis://127.0.0.1:{}", primary.port),
            Duration::from_millis(600),
            Duration::from_millis(200),
        )
        .await
        .expect("startup must allow the optional replica response to time out");
        let exists: bool = connection
            .exists("test-key")
            .await
            .expect("healthy primary");
        expect_that!(exists, eq(true));
        expect_that!(
            replica.redirect.readonly_requests.load(Ordering::SeqCst),
            gt(0)
        );
    }

    #[gtest]
    #[tokio::test]
    async fn recovers_with_unresponsive_replica_and_healthy_primary() {
        let topology = Arc::new(AtomicU16::new(0));
        let discoveries = Arc::new(AtomicUsize::new(0));
        let writes = Arc::new(AtomicUsize::new(0));
        let old = node(topology.clone(), discoveries.clone(), writes.clone()).await;
        let primary = node(topology.clone(), discoveries.clone(), writes.clone()).await;
        let replica = node(topology.clone(), discoveries.clone(), writes.clone()).await;
        let seed = node(topology.clone(), discoveries.clone(), writes.clone()).await;
        topology.store(old.port, Ordering::SeqCst);
        let mut connection = RecoveringClusterConnection::with_timeouts(
            format!("redis://127.0.0.1:{}", seed.port),
            Duration::from_millis(600),
            Duration::from_millis(200),
        )
        .await
        .expect("initial healthy topology");

        topology.store(primary.port, Ordering::SeqCst);
        seed.redirect.replica.store(replica.port, Ordering::SeqCst);
        replica
            .redirect
            .stall_readonly
            .store(true, Ordering::SeqCst);
        old.blackhole.store(true, Ordering::SeqCst);
        let result: RedisResult<bool> = connection.exists("test-key").await;
        expect_that!(result.is_err(), eq(true));
        timeout(Duration::from_secs(3), async {
            loop {
                if matches!(connection.exists::<_, bool>("test-key").await, Ok(true)) {
                    break;
                }
                sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("recovery must skip the unresponsive replica and use the healthy primary");
        expect_that!(
            replica.redirect.readonly_requests.load(Ordering::SeqCst),
            gt(0)
        );
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
    async fn redirect_fixture() -> (RecoveringClusterConnection, Node, Node, Arc<AtomicUsize>) {
        let topology = Arc::new(AtomicU16::new(0));
        let discoveries = Arc::new(AtomicUsize::new(0));
        let writes = Arc::new(AtomicUsize::new(0));
        let old = node(topology.clone(), discoveries.clone(), writes.clone()).await;
        let target = node(topology.clone(), discoveries, writes.clone()).await;
        topology.store(old.port, Ordering::SeqCst);
        let conn = RecoveringClusterConnection::with_timeouts(
            format!("redis://127.0.0.1:{}", old.port),
            Duration::from_millis(500),
            Duration::from_millis(500),
        )
        .await
        .expect("initial cluster connection");
        old.redirect.target.store(target.port, Ordering::SeqCst);
        (conn, old, target, writes)
    }

    #[gtest]
    #[tokio::test]
    async fn ask_follows_target_without_changing_slot_topology() {
        let (mut conn, _old, target, writes) = redirect_fixture().await;
        target.redirect.require_asking.store(true, Ordering::SeqCst);
        for _ in 0..2 {
            let result: i64 = conn
                .incr("counter", 1)
                .await
                .expect("ASK followed with ASKING");
            expect_that!(result, eq(1));
        }
        expect_that!(target.redirect.asking_count.load(Ordering::SeqCst), eq(2));
        expect_that!(
            writes.load(Ordering::SeqCst),
            eq(2),
            "execute each write exactly once"
        );
        expect_that!(
            conn.shared.state.read().await.generation,
            eq(0),
            "ASK does not require reseeding"
        );
    }

    #[gtest]
    #[tokio::test]
    async fn moved_follows_target_without_replaying_writes_on_old_node() {
        let (mut conn, old, target, writes) = redirect_fixture().await;
        old.redirect.moved.store(true, Ordering::SeqCst);
        let _: i64 = conn.incr("counter", 1).await.expect("MOVED followed");
        expect_that!(writes.load(Ordering::SeqCst), eq(1));
        expect_that!(target.redirect.asking_count.load(Ordering::SeqCst), eq(0));
    }

    #[gtest]
    #[tokio::test]
    async fn redirected_write_timeout_is_not_replayed() {
        let (mut conn, _old, target, writes) = redirect_fixture().await;
        target.redirect.require_asking.store(true, Ordering::SeqCst);
        target.redirect.stall_writes.store(true, Ordering::SeqCst);
        let result: RedisResult<i64> = conn.incr("counter", 1).await;
        expect_that!(
            result.expect_err("ambiguous write times out").is_timeout(),
            eq(true)
        );
        expect_that!(writes.load(Ordering::SeqCst), eq(1));
    }

    #[gtest]
    #[tokio::test]
    async fn redirect_loop_is_bounded() {
        let (mut conn, old, target, writes) = redirect_fixture().await;
        target.redirect.target.store(old.port, Ordering::SeqCst);
        let result: RedisResult<i64> = conn.incr("counter", 1).await;
        expect_that!(
            result.expect_err("redirect budget exhausted").code(),
            some(eq("ASK"))
        );
        expect_that!(
            old.redirect.requests.load(Ordering::SeqCst)
                + target.redirect.requests.load(Ordering::SeqCst),
            eq(MAX_REDIRECTS + 1)
        );
        expect_that!(writes.load(Ordering::SeqCst), eq(0));
    }
}

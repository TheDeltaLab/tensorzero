// Modified by Delta-AI under Apache 2.0
use crate::error::ErrorDetails;
use crate::observability::{LogFormat, TENSORZERO_EMBEDDED_DEFAULTS, setup_observability};

use super::ValkeyConnectionInfo;
use crate::db::{ConsumeTicketsRequest, RateLimitQueries};
use crate::rate_limiting::{ActiveRateLimitKey, RateLimitInterval};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::tcp::OwnedReadHalf;
use tokio::net::{TcpListener, TcpStream};

/// Test that connecting with a TLS URL (`rediss://`) produces a connection error,
/// not a rustls crypto provider panic. This validates that `setup_observability`
/// correctly installs the rustls crypto provider so that the `redis` crate's
/// TLS support works.
#[tokio::test]
async fn test_tls_url_gives_connection_error() {
    setup_observability(LogFormat::Pretty, TENSORZERO_EMBEDDED_DEFAULTS)
        .await
        .unwrap();

    let result = ValkeyConnectionInfo::new("rediss://tensorzero.invalid:6379").await;
    let err = result
        .err()
        .expect("TLS connection to non-TLS server should fail");
    assert!(
        matches!(err.get_details(), ErrorDetails::ValkeyConnection { .. }),
        "expected ValkeyConnection error, got: {err}"
    );
}

#[tokio::test]
async fn test_tls_url_gives_connection_error_cache_only() {
    setup_observability(LogFormat::Pretty, TENSORZERO_EMBEDDED_DEFAULTS)
        .await
        .unwrap();

    let result = ValkeyConnectionInfo::new_cache_only("rediss://tensorzero.invalid:6379").await;
    let err = result
        .err()
        .expect("TLS connection to non-TLS server should fail");
    assert!(
        matches!(err.get_details(), ErrorDetails::ValkeyConnection { .. }),
        "expected ValkeyConnection error, got: {err}"
    );
}

#[test]
fn test_strip_cluster_fragment() {
    use super::strip_cluster_fragment;
    // Plain URL: unchanged, no cluster.
    assert_eq!(
        strip_cluster_fragment("redis://127.0.0.1:6379"),
        ("redis://127.0.0.1:6379".to_string(), false)
    );
    // Cluster flag: stripped, requested.
    assert_eq!(
        strip_cluster_fragment("rediss://:secret@host:8500#cluster"),
        ("rediss://:secret@host:8500".to_string(), true)
    );
    // redis-rs `insecure` flag is preserved on its own.
    assert_eq!(
        strip_cluster_fragment("rediss://u:p@host:6379#insecure"),
        ("rediss://u:p@host:6379#insecure".to_string(), false)
    );
    // Combined flags: cluster removed, insecure kept.
    assert_eq!(
        strip_cluster_fragment("rediss://u:p@host:8500#insecure+cluster"),
        ("rediss://u:p@host:8500#insecure".to_string(), true)
    );
}

/// Read one RESP command (`*<n> ... $<len> ...`) from a connection.
async fn read_resp_command(reader: &mut BufReader<OwnedReadHalf>) -> Option<Vec<String>> {
    let mut line = String::new();
    reader.read_line(&mut line).await.ok()?;
    let count: usize = line.strip_prefix('*')?.trim().parse().ok()?;
    let mut args = Vec::with_capacity(count);
    for _ in 0..count {
        line.clear();
        reader.read_line(&mut line).await.ok()?;
        let size: usize = line.strip_prefix('$')?.trim().parse().ok()?;
        let mut data = vec![0u8; size + 2];
        reader.read_exact(&mut data).await.ok()?;
        args.push(String::from_utf8(data[..size].to_vec()).ok()?);
    }
    Some(args)
}

/// Behavior for a fake cluster node in the rate-limiting MOVED test.
struct ClusterNodeBehavior {
    /// Port advertised as the owner of all slots in `CLUSTER SLOTS`.
    advertised_owner: u16,
    /// When set, answer the consume `FCALL` with a `MOVED` redirect to this
    /// port instead of serving it (models a shard that does not own the key).
    redirect_consume_to: Option<u16>,
}

/// A fake cluster node: answers topology discovery and the function-library
/// load, then either serves the rate-limit `FCALL` or redirects it to another
/// node, exercising the MOVED-following path.
async fn serve_cluster_node(stream: TcpStream, behavior: ClusterNodeBehavior) {
    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);
    while let Some(args) = read_resp_command(&mut reader).await {
        let reply = match args.first().map(String::as_str).unwrap_or("") {
            "PING" => "+PONG\r\n".to_string(),
            "FUNCTION" => "+OK\r\n".to_string(),
            "CLUSTER" => {
                // A single shard owning every slot, advertised at `advertised_owner`.
                let port = behavior.advertised_owner;
                format!("*1\r\n*3\r\n:0\r\n:16383\r\n*2\r\n$9\r\n127.0.0.1\r\n:{port}\r\n")
            }
            "FCALL" => match args.get(1).map(String::as_str).unwrap_or("") {
                "tensorzero_consume_tickets_v2" => match behavior.redirect_consume_to {
                    Some(port) => format!("-MOVED 123 127.0.0.1:{port}\r\n"),
                    None => {
                        let key = args.get(3).map(String::as_str).unwrap_or("");
                        let body = format!(
                            "[{{\"key\":\"{key}\",\"success\":true,\"remaining\":9,\"consumed\":1}}]"
                        );
                        format!("${}\r\n{}\r\n", body.len(), body)
                    }
                },
                _ => {
                    let body = "{\"migrated_count\":0}";
                    format!("${}\r\n{}\r\n", body.len(), body)
                }
            },
            _ => "+OK\r\n".to_string(),
        };
        if writer.write_all(reply.as_bytes()).await.is_err() {
            return;
        }
    }
}

/// Regression test for the "Azure rate-limit Redis MOVED" failure.
///
/// Azure Managed Valkey runs in cluster mode. A seed shard answers `MOVED` for
/// a rate-limit key it does not own; with the `#cluster` URL flag the gateway
/// must follow the redirect (through a cluster-aware connection) and consume
/// tickets on the owning shard instead of failing the request closed (502).
#[tokio::test]
#[expect(
    clippy::disallowed_methods,
    reason = "Fake cluster listeners are torn down with the test runtime"
)]
async fn test_rate_limiting_follows_moved_on_cluster() {
    // The target shard actually serves the consume `FCALL`.
    let target_listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind target node");
    let target_port = target_listener.local_addr().expect("bound address").port();
    let target = tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = target_listener.accept().await else {
                break;
            };
            tokio::spawn(serve_cluster_node(
                stream,
                ClusterNodeBehavior {
                    advertised_owner: target_port,
                    redirect_consume_to: None,
                },
            ));
        }
    });

    // The seed advertises itself as owner of all slots but redirects the
    // consume `FCALL` to the target, as a real shard does for a foreign key.
    let seed_listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind seed node");
    let seed_port = seed_listener.local_addr().expect("bound address").port();
    let seed = tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = seed_listener.accept().await else {
                break;
            };
            tokio::spawn(serve_cluster_node(
                stream,
                ClusterNodeBehavior {
                    advertised_owner: seed_port,
                    redirect_consume_to: Some(target_port),
                },
            ));
        }
    });

    let info = ValkeyConnectionInfo::new(&format!("redis://127.0.0.1:{seed_port}#cluster"))
        .await
        .expect("gateway connects to the fake cluster seed");

    let requests = vec![ConsumeTicketsRequest {
        key: ActiveRateLimitKey::new("tensorzero_ratelimit:test".to_string()),
        requested: 1,
        capacity: 10,
        refill_amount: 1,
        refill_interval: RateLimitInterval::Minute,
    }];

    let receipts = info
        .consume_tickets(&requests)
        .await
        .expect("rate limiting should follow the MOVED redirect and succeed");
    assert_eq!(receipts.len(), 1, "one receipt per request");
    assert!(receipts[0].success, "consume should succeed after redirect");
    assert_eq!(receipts[0].tickets_consumed, 1);

    seed.abort();
    target.abort();
}

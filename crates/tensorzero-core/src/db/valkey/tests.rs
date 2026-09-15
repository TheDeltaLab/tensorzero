// Modified by Delta-AI under Apache 2.0
use crate::error::ErrorDetails;
use crate::observability::{LogFormat, TENSORZERO_EMBEDDED_DEFAULTS, setup_observability};

use super::ValkeyConnectionInfo;

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

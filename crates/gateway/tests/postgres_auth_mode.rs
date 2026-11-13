// Modified by Delta-AI under Apache 2.0
//! Exercise the actual CLI routes so new direct SQLx connections cannot bypass auth configuration.
use std::process::Command;

use googletest::prelude::*;

#[gtest]
fn postgres_commands_and_database_config_startup_validate_auth_mode() {
    let directory = tempfile::tempdir().expect("temporary config directory");
    let config = directory.path().join("tensorzero.toml");
    std::fs::write(
        &config,
        "[gateway]\ndisable_pseudonymous_usage_analytics = true\n",
    )
    .expect("write minimal config");
    let config = config.to_str().expect("UTF-8 temporary path");
    let commands = [
        vec![], // Explicitly opt in to loading startup configuration from Postgres.
        vec!["--migrate-config", config],
        vec!["--run-postgres-migrations"],
        vec!["--create-api-key"],
        vec!["--import-synapse-api-key", "test-key"],
        vec!["--disable-api-key", "test-public-id"],
    ];
    for arguments in commands {
        let output = Command::new(env!("CARGO_BIN_EXE_gateway"))
            .args(&arguments)
            .env_remove("TENSORZERO_CONFIG_FILE")
            .env("TENSORZERO_INTERNAL_FLAG_ENABLE_CONFIG_IN_DATABASE", "true")
            .env(
                "TENSORZERO_POSTGRES_URL",
                "postgres://test@127.0.0.1:1/test",
            )
            .env("TENSORZERO_POSTGRES_AUTH_MODE", "invalid-test-mode")
            .env("RUST_LOG", "error")
            .output()
            .expect("run gateway command");
        let logs = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        expect_that!(
            output.status.success(),
            eq(false),
            "command {arguments:?} must fail"
        );
        expect_that!(
            logs,
            contains_substring("TENSORZERO_POSTGRES_AUTH_MODE must be"),
            "command {arguments:?} must reach the shared auth validation, output: {logs}"
        );
    }
}

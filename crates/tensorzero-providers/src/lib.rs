// Modified by Delta-AI under Apache 2.0
// This is an internal crate, so we're the only consumers of
// traits with async fns for now.
#![expect(async_fn_in_trait)]
#![recursion_limit = "256"]

//! Model provider implementations for TensorZero, extracted from `tensorzero-core`.
//!
//! This crate owns:
//! * `providers` — one module per provider plus shared helpers
//! * `provider_config` — the `ProviderConfig` / `UninitializedProviderConfig`
//!   enums, their `load` logic, and the per-provider dispatch methods
//! * `provider_types` — the `[providers]` config-file section
//!   (`ProviderTypesConfig`)
//! * `default_credentials` — `ProviderTypeDefaultCredentials` (lazy per-provider
//!   default credentials)
//! * `embedding_provider` — the `EmbeddingProviderConfig` enum
//! * `routing`, `observability_tags`, `throughput_tracker` — request-routing /
//!   observability state used by `providers::helpers` (re-exported by
//!   `tensorzero-core` for its endpoints)
//! * `jsonschema_util` — JSON schema wrapper used by provider tests and config
//!   validation (re-exported by `tensorzero-core`)
//!
//! `tensorzero-core` re-exports everything, so downstream crates should not
//! need to depend on this crate directly.

pub mod default_credentials;
pub mod embedding_provider;
pub mod jsonschema_util;
pub mod observability_tags;
pub mod provider_config;
pub mod provider_types;
pub mod providers;
pub mod routing;
pub mod throughput_tracker;

#[cfg(any(test, feature = "e2e_tests"))]
pub mod utils {
    //! Test/e2e utilities (moved from `tensorzero-core`).
    pub mod testing;
}

#[cfg(any(test, feature = "test-helpers"))]
pub mod test_utils {
    //! Cross-crate test fixtures. `tensorzero-core` enables the `test-helpers`
    //! feature in its dev-dependencies so its tests can reach
    //! `providers::test_helpers`.
    pub use crate::providers::test_helpers;
}

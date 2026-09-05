# MODIFICATIONS.md

This file tracks modifications to **non-source-code files** in this fork.

Source code files (`.rs`, `.ts`, `.tsx`, `.py`, `.js`, `.jsx`, `.css`, `.scss`, `.sql`, `.sh`) carry per-file modification notices in their headers and are **not** listed here.

See `NOTICE` for the overall attribution statement.

---

## Modified non-source-code files

- `AGENTS.md` — Added Modification Notice (Delta-AI fork) section.
- `CLA.md` — Changed Company from TensorZero, Inc. to Delta-AI; removed legacy hello@tensorzero.com contact.
- `SECURITY.md` — Changed security contact to security@thebrainly.ai.
- `crates/tensorzero-stored-config/src/postgres/migrations/20260622000001_model_aliases.sql` — New model_aliases DB migration table (Delta-AI fork).
- `docs/synapse-migration-plan.md` — Synapse → TensorZero migration plan (Delta-AI fork).
- `docs/superpowers/plans/2026-06-22-model-alias.md` — Model alias implementation plan (Delta-AI fork).
- `.github/workflows/general.yml` — Changed lint-rust from 4-partition `cargo hack --each-feature` to single `cargo clippy --all-features` (Delta-AI fork).
- `.github/workflows/publish-ghcr.yml` — Publish gateway and UI images to GHCR for this fork; runs on self-hosted `tensorzero-ci` runner with host-disk buildx layer cache under `/mnt/runner/buildx-cache` and a persistent named builder (Delta-AI fork).
- `crates/gateway/Dockerfile` — Default container bind address `0.0.0.0:3720`; cargo registry/target BuildKit cache mounts for faster rebuilds (Delta-AI fork).
- `ui/Dockerfile` — Default UI listen port `3721`; cargo registry/target BuildKit cache mounts in the tensorzero-node build stage (Delta-AI fork).
- `crates/tensorzero-core/tests/e2e/config/tensorzero.model_aliases.toml` — E2E alias failover fixtures (Delta-AI fork).
- `examples/guides/synapse-compat/config/tensorzero.toml` — Synapse-compatible `[model_aliases]` for public providers (Delta-AI fork).
- `crates/Cargo.lock` — Workspace lockfile for Synapse-compat auth (bcrypt) and HTTP timeout deps (Delta-AI fork).
- `.github/workflows/modification-notice-check.yml` — Exclude generated ts-rs bindings from header notice check (Delta-AI fork).
- `crates/Cargo.toml` — Added chrono-tz for peak/off-peak cost windows (Delta-AI fork).
- `crates/tensorzero-core/Cargo.toml` — Added chrono-tz for peak/off-peak cost windows (Delta-AI fork).
- `docs/operations/track-usage-and-cost.mdx` — Documented per-provider usage pointers, peak windows, pointer lists, token-length tiers, currency, and tag filters (Delta-AI fork).
- `docs/gateway/api-reference/inference-openai-compatible.mdx` — Documented `x-tensorzero-*` header aliases, episode-id header, and `x-tensorzero-tags` (Delta-AI fork).
- `docs/gateway/configuration-reference.mdx` — Documented `usage`, `peak`, `timezone`, `currency`, pointer lists, and token-length tiers (Delta-AI fork).
- `examples/docs/guides/operations/track-usage-and-cost/config/tensorzero.toml` — Peak/off-peak, GLM-5.1-style bucket cost, and CNY examples (Delta-AI fork).
- `crates/tensorzero-http/Cargo.toml` — Added tokio for per-request timeout override (Delta-AI fork).
- `crates/tensorzero-auth/Cargo.toml` — Added bcrypt for imported Synapse API keys (Delta-AI fork).
- `crates/tensorzero-core/tests/load/synapse-compat/tensorzero.toml` — Dummy-backed Synapse-compat load-test gateway config, Postgres observability (Delta-AI fork).
- `crates/tensorzero-core/tests/load/synapse-compat/tensorzero.no-obs.toml` — Same load-test config with observability off (Delta-AI fork).
- `crates/tensorzero-core/tests/load/synapse-compat/tensorzero.postgres.batch.toml` — Postgres batch-write variant of the Synapse-compat load-test config (Delta-AI fork).
- `crates/tensorzero-core/tests/load/synapse-compat/docker-compose.yml` — Dedicated Postgres 16 on host port 5433 for Synapse-compat load tests (Delta-AI fork).
- `crates/tensorzero-core/tests/load/synapse-compat/README.md` — How to run Synapse-compat vegeta concurrency sweeps (Delta-AI fork).
- `crates/tensorzero-core/tests/load/synapse-compat/bodies/chat.json` — Load-test body (Delta-AI fork).
- `crates/tensorzero-core/tests/load/synapse-compat/bodies/chat-stream.json` — Load-test body (Delta-AI fork).
- `crates/tensorzero-core/tests/load/synapse-compat/bodies/chat-long-stream.json` — 40k-thinking + 10k-text streaming chat body (Delta-AI fork).
- `crates/tensorzero-core/tests/load/synapse-compat/bodies/messages.json` — Load-test body (Delta-AI fork).
- `crates/tensorzero-core/tests/load/synapse-compat/bodies/embeddings.json` — Load-test body (Delta-AI fork).
- `crates/tensorzero-core/tests/load/synapse-compat/bodies/rerank.json` — Load-test body (Delta-AI fork).
- `crates/tensorzero-core/tests/load/synapse-compat/bodies/completions.json` — Load-test body (Delta-AI fork).
- `crates/tensorzero-core/tests/load/synapse-compat/bodies/responses.json` — Load-test body (Delta-AI fork).
- `.github/workflows/codeql.yml` — Disabled automatic push/PR/scheduled triggers; manual `workflow_dispatch` only (Delta-AI fork).

- `crates/Cargo.toml` — Added `async-inference` workspace member for the async inference API worker (Delta-AI fork).
- `crates/Cargo.lock` — Workspace lockfile updated for the `async-inference` crate and its deps (Delta-AI fork).
- `crates/async-inference/Cargo.toml` — New crate manifest for the async inference durable worker (Delta-AI fork).
- `crates/durable-tools-spawn/Cargo.toml` — Added chrono for task timing reads used by the async inference status endpoint (Delta-AI fork).
- `crates/gateway/Cargo.toml` — Added `async-inference` dependency for the embedded async inference worker (Delta-AI fork).
- `crates/tensorzero-core/tests/e2e/config/async-inference.gateway.toml` — E2E config override enabling `[gateway.async_inference]` for the async inference API tests (Delta-AI fork).
- `crates/.sqlx/query-163a79b376e88675aa684f9d7c9ece4dbcce1e739931950d83e5bb63b9cc7337.json` — Regenerated sqlx offline cache for inference storage/protection queries (Delta-AI fork).
- `crates/.sqlx/query-203c3c5c22d937daa6e6e87e1ed5bfd03f309b943b9f60ee3187e57d76cb80c6.json` — Regenerated sqlx offline cache for inference storage/protection queries (Delta-AI fork).
- `crates/.sqlx/query-3eb37353d1fed4eddc84bfb252ab9e998911167292de1afdea4cc82266bf918b.json` — Regenerated sqlx offline cache for inference storage/protection queries (Delta-AI fork).
- `crates/.sqlx/query-5f9e9a43f47eff49fa1b9db80fa1e61b192425c3315b37fca58ea3be58ee021e.json` — Regenerated sqlx offline cache for inference storage/protection queries (Delta-AI fork).
- `crates/.sqlx/query-6c68ff37b0cceda3789e94dbf8c7a34d077f5f852b88d4d6ba82c4c7bd3b266a.json` — Regenerated sqlx offline cache for inference storage/protection queries (Delta-AI fork).
- `crates/.sqlx/query-c285d5cf693d465e61454a15bc0512656418e71fbadb4a905ab401900d417b95.json` — Regenerated sqlx offline cache for inference storage/protection queries (Delta-AI fork).
- `crates/.sqlx/query-d52d769116fa2588c6a1078dd9522333e3824c604cf94fb442e686edc7033c52.json` — Regenerated sqlx offline cache for inference storage/protection queries (Delta-AI fork).
- `crates/.sqlx/query-e5a7e556f0c133cddff0613af6d60cf7f13c688d9f80f6aa92112dcb9796e1a2.json` — Regenerated sqlx offline cache for inference storage/protection queries (Delta-AI fork).
- `.github/workflows/general.yml` — Gate lint-rust/rust-build/rust-test/validate-node/validate-python on language-specific path filters in detect-changes (Delta-AI fork).

---

_To add an entry: append a bullet item above. The CI workflow (`modification-notice-check`) will verify that every non-source-code file modified in a PR is listed here._

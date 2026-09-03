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
- `.github/workflows/publish-ghcr.yml` — Publish gateway and UI images to GHCR for this fork (Delta-AI fork).
- `crates/gateway/Dockerfile` — Default container bind address `0.0.0.0:3720` (Delta-AI fork).
- `ui/Dockerfile` — Default UI listen port `3721` (Delta-AI fork).
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

- `crates/Cargo.toml` — Added `async-inference` workspace member for the async inference API worker (Delta-AI fork).
- `crates/Cargo.lock` — Workspace lockfile updated for the `async-inference` crate and its deps (Delta-AI fork).
- `crates/async-inference/Cargo.toml` — New crate manifest for the async inference durable worker (Delta-AI fork).
- `crates/durable-tools-spawn/Cargo.toml` — Added chrono for task timing reads used by the async inference status endpoint (Delta-AI fork).
- `crates/gateway/Cargo.toml` — Added `async-inference` dependency for the embedded async inference worker (Delta-AI fork).

---

_To add an entry: append a bullet item above. The CI workflow (`modification-notice-check`) will verify that every non-source-code file modified in a PR is listed here._

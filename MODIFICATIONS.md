# MODIFICATIONS.md

This file tracks modifications to **non-source-code files** in this fork.

Source code files (`.rs`, `.ts`, `.tsx`, `.py`, `.js`, `.jsx`, `.css`, `.scss`, `.sql`, `.sh`) carry per-file modification notices in their headers and are **not** listed here.

See `NOTICE` for the overall attribution statement.

---

## Modified non-source-code files

- `AGENTS.md` — Added Modification Notice (Delta-AI fork) section.
- `AGENTS.md` — Require `CARGO_TARGET_DIR=~/.tensorzero-cargo-dir` for all `cargo` commands so checkouts/worktrees share one build cache (Delta-AI fork).
- `crates/.config/nextest.toml` — Longer slow-timeout override for the env-gated live-gateway async inference e2e tests (`client::async_inference::tests::e2e`) (Delta-AI fork).
- `crates/tensorzero-python/tensorzero/tensorzero.pyi` — Type stubs for the async task / status / health gateway methods (Delta-AI fork).
- `CLA.md` — Changed Company from TensorZero, Inc. to Delta-AI; removed legacy hello@tensorzero.com contact.
- `SECURITY.md` — Changed security contact to security@thebrainly.ai.
- `crates/tensorzero-stored-config/src/postgres/migrations/20260622000001_model_aliases.sql` — New model_aliases DB migration table (Delta-AI fork).
- `docs/synapse-migration-plan.md` — Synapse → TensorZero migration plan (Delta-AI fork).
- `docs/superpowers/plans/2026-06-22-model-alias.md` — Model alias implementation plan (Delta-AI fork).
- `.github/workflows/general.yml` — Changed lint-rust from 4-partition `cargo hack --each-feature` to single `cargo clippy --all-features` (Delta-AI fork).
- `.github/workflows/publish-ghcr.yml` — Publish gateway and UI images to GHCR for this fork; runs on self-hosted `tensorzero-ci` runner with host-disk buildx layer cache under `/mnt/runner/buildx-cache` and a persistent named builder (Delta-AI fork). Gateway LTO level is selectable: `push` builds use plain `release` (no LTO), release builds use thin LTO, and `workflow_dispatch` accepts `lto` (none/thin/fat) and `ref` inputs; non-default LTO builds get `-thin`/`-fat` image tag suffixes. A 03:00 Asia/Shanghai schedule backfills a fat-LTO image for the latest release when missing.
- `crates/gateway/Dockerfile` — Default container bind address `0.0.0.0:3720`; cargo registry/target BuildKit cache mounts for faster rebuilds; `CARGO_BUILD_JOBS=4` default to cap parallelism on the shared CI runner; `PROFILE` defaults to `release` (no LTO) so ad-hoc local builds stay fast — CI passes an explicit `PROFILE` for thin/fat LTO images (Delta-AI fork).
- `ui/Dockerfile` — Default UI listen port `3721`; cargo registry/target BuildKit cache mounts in the tensorzero-node build stage; `CARGO_BUILD_JOBS=4` default to cap parallelism on the shared CI runner (Delta-AI fork).
- `crates/tensorzero-core/tests/e2e/config/tensorzero.model_aliases.toml` — E2E alias failover fixtures (Delta-AI fork).
- `examples/guides/synapse-compat/config/tensorzero.toml` — Synapse-compatible `[model_aliases]` for public providers (Delta-AI fork).
- `crates/Cargo.lock` — Workspace lockfile for Synapse-compat auth (bcrypt) and HTTP timeout deps (Delta-AI fork).
- `.github/workflows/modification-notice-check.yml` — Exclude generated ts-rs bindings from header notice check (Delta-AI fork).
- `crates/Cargo.toml` — Added chrono-tz for peak/off-peak cost windows (Delta-AI fork).
- `crates/Cargo.toml` — Added `profile.thin` (thin LTO release build) for faster GHCR image builds (Delta-AI fork).
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
- `crates/Cargo.toml` — Enabled redis `tls-rustls-insecure` feature so `rediss://...#insecure` URLs work for Aliyun Tair, whose TLS cert is signed by an internal CA (Delta-AI fork).
- `crates/.sqlx/query-163a79b376e88675aa684f9d7c9ece4dbcce1e739931950d83e5bb63b9cc7337.json` — Regenerated sqlx offline cache for inference storage/protection queries (Delta-AI fork).
- `crates/.sqlx/query-203c3c5c22d937daa6e6e87e1ed5bfd03f309b943b9f60ee3187e57d76cb80c6.json` — Regenerated sqlx offline cache for inference storage/protection queries (Delta-AI fork).
- `crates/.sqlx/query-3eb37353d1fed4eddc84bfb252ab9e998911167292de1afdea4cc82266bf918b.json` — Regenerated sqlx offline cache for inference storage/protection queries (Delta-AI fork).
- `crates/.sqlx/query-5f9e9a43f47eff49fa1b9db80fa1e61b192425c3315b37fca58ea3be58ee021e.json` — Regenerated sqlx offline cache for inference storage/protection queries (Delta-AI fork).
- `crates/.sqlx/query-6c68ff37b0cceda3789e94dbf8c7a34d077f5f852b88d4d6ba82c4c7bd3b266a.json` — Regenerated sqlx offline cache for inference storage/protection queries (Delta-AI fork).
- `crates/.sqlx/query-6a3696a3de56d6051d1d47a9bb6748a8d640914b43c1ab07ca5b8e6a0a8ecef3.json` — Regenerated sqlx offline cache for the failed-inference `error` column queries (Delta-AI fork).
- `crates/.sqlx/query-cd05554b23fc43577847556dadd63ce1d9bc9bcbd0b1d61665abcee866c9fd2a.json` — Regenerated sqlx offline cache for the failed-inference `error` column queries (Delta-AI fork).
- `crates/.sqlx/query-c285d5cf693d465e61454a15bc0512656418e71fbadb4a905ab401900d417b95.json` — Regenerated sqlx offline cache for inference storage/protection queries (Delta-AI fork).
- `crates/.sqlx/query-d52d769116fa2588c6a1078dd9522333e3824c604cf94fb442e686edc7033c52.json` — Regenerated sqlx offline cache for inference storage/protection queries (Delta-AI fork).
- `crates/.sqlx/query-e5a7e556f0c133cddff0613af6d60cf7f13c688d9f80f6aa92112dcb9796e1a2.json` — Regenerated sqlx offline cache for inference storage/protection queries (Delta-AI fork).
- `.github/workflows/general.yml` — Gate lint-rust/rust-build/rust-test/validate-node/validate-python on language-specific path filters in detect-changes (Delta-AI fork).

---

_To add an entry: append a bullet item above. The CI workflow (`modification-notice-check`) will verify that every non-source-code file modified in a PR is listed here._

- `crates/Cargo.toml` — Added `tensorzero-providers` workspace member (Delta-AI fork).
- `crates/Cargo.lock` — Workspace lockfile updated for the `tensorzero-providers` crate (Delta-AI fork).
- `crates/tensorzero-providers/Cargo.toml` — New crate: model provider implementations extracted from `tensorzero-core` so provider-only changes don't recompile all of core (Delta-AI fork).
- `crates/tensorzero-core/Cargo.toml` — Depends on `tensorzero-providers`; forwards the `e2e_tests`/`pyo3` features and enables `test-helpers` in dev-dependencies (Delta-AI fork).
- `crates/tensorzero-optimizers/Cargo.toml` — Depends on `tensorzero-providers` for GCP Vertex Gemini fine-tuning API types (Delta-AI fork).
- `crates/tensorzero-core/src/providers/AGENTS.md` — Moved to `crates/tensorzero-providers/src/providers/AGENTS.md` with the providers split (Delta-AI fork).
- `crates/tensorzero-core/src/providers/CLAUDE.md` — Moved to `crates/tensorzero-providers/src/providers/CLAUDE.md` with the providers split (Delta-AI fork).
- `sdk/go/client.go` — New Go SDK: async-task HTTP client (submit/getTask) (Delta-AI fork).
- `sdk/go/errors.go` — New Go SDK: typed errors (TaskNotFoundError, HTTPError) (Delta-AI fork).
- `sdk/go/sse.go` — New Go SDK: SSE parser over the raw response body (Delta-AI fork).
- `sdk/go/stream.go` — New Go SDK: wire-faithful StreamTask with reconnect/dedup and 410 poll fallback (Delta-AI fork).
- `sdk/go/tasks.go` — New Go SDK: TaskStatus discriminated union types (Delta-AI fork).
- `sdk/go/wait.go` — New Go SDK: WaitForCompletion exponential-backoff polling (Delta-AI fork).
- `sdk/go/sse_test.go` — New Go SDK: SSE parser unit tests (Delta-AI fork).
- `sdk/go/stream_test.go` — New Go SDK: StreamTask unit tests (Delta-AI fork).
- `sdk/go/tasks_test.go` — New Go SDK: task status parsing unit tests (Delta-AI fork).
- `sdk/go/wait_test.go` — New Go SDK: WaitForCompletion unit tests (Delta-AI fork).
- `sdk/go/e2e_test.go` — New Go SDK: env-gated e2e tests (TZ_E2E_GATEWAY + TZ_E2E_KEY required) (Delta-AI fork).
- `sdk/go/go.mod` — New Go SDK submodule `github.com/TheDeltaLab/tensorzero/sdk/go` (Delta-AI fork).
- `sdk/go/README.md` — New Go SDK README (Delta-AI fork).
- `sdk/typescript/.gitignore` — New TypeScript SDK workspace gitignore (node_modules/dist) (Delta-AI fork).
- `sdk/typescript/package.json` — New TypeScript SDK private workspace root (Delta-AI fork).
- `sdk/typescript/pnpm-workspace.yaml` — New TypeScript SDK pnpm workspace (Delta-AI fork).
- `sdk/typescript/pnpm-lock.yaml` — New TypeScript SDK lockfile (Delta-AI fork).
- `sdk/typescript/tsconfig.base.json` — New TypeScript SDK shared tsconfig (Delta-AI fork).
- `sdk/typescript/README.md` — New TypeScript SDK README (Delta-AI fork).
- `sdk/typescript/packages/tensorzero-sdk/package.json` — New package `@thedeltalab/tensorzero-sdk`, published to GitHub Packages (Delta-AI fork).
- `sdk/typescript/packages/tensorzero-sdk/tsconfig.json` — New package tsconfig (Delta-AI fork).
- `sdk/typescript/packages/ai-sdk-provider/package.json` — New package `@thedeltalab/ai-sdk-provider`, published to GitHub Packages (Delta-AI fork).
- `sdk/typescript/packages/ai-sdk-provider/tsconfig.json` — New package tsconfig (Delta-AI fork).
- `.github/workflows/publish-sdk-typescript.yml` — Publish `@delta-ai/tensorzero-sdk` and `@delta-ai/ai-sdk-provider` to npmjs via OIDC trusted publishing (`npm-release` environment) on `sdk/ts/v*` tags (Delta-AI fork).
- `crates/Cargo.toml` — Added `opentelemetry-appender-tracing` for OTLP logs export (Delta-AI fork).
- `crates/tensorzero-otel/Cargo.toml` — Added `opentelemetry-appender-tracing` dependency for the tracing-to-OTLP logs bridge (Delta-AI fork).
- `crates/Cargo.lock` — Workspace lockfile updated for `opentelemetry-appender-tracing` (Delta-AI fork).
- `docs/operations/export-opentelemetry-logs.mdx` — Documented `gateway.export.otlp.logs` OTLP logs export (Delta-AI fork).
- `docs/docs.json` — Registered the `operations/export-opentelemetry-logs` nav entry (Delta-AI fork).

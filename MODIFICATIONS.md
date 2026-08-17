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

---

_To add an entry: append a bullet item above. The CI workflow (`modification-notice-check`) will verify that every non-source-code file modified in a PR is listed here._

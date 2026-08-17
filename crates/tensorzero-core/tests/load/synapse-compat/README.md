# Synapse-compat load tests

Closed-loop concurrency sweeps against Synapse-compatible gateway routes. The upstream LLM is the Dummy provider (`dummy::*`), so the numbers isolate gateway + Postgres observability cost, not vendor latency.

Observability uses **Postgres as primary** (`gateway.observability.backend = "postgres"`). Do not set `TENSORZERO_CLICKHOUSE_URL`.

## Setup

1. Postgres (or disable observability; see below):

   ```bash
   docker compose -f crates/tensorzero-core/tests/load/synapse-compat/docker-compose.yml up postgres --wait
   export TENSORZERO_POSTGRES_URL=postgres://postgres:postgres@127.0.0.1:5433/tensorzero_load
   ```

   Host port `5433` avoids colliding with other local Postgres stacks on `5432`.

2. Run migrations once, then start the gateway with Dummy enabled (`e2e_tests` feature):

   ```bash
   cd crates
   cargo run --release -p gateway --features e2e_tests -- --run-postgres-migrations
   cargo run --release -p gateway --features e2e_tests -- \
     --config-file tensorzero-core/tests/load/synapse-compat/tensorzero.toml \
     --bind-address 127.0.0.1:3000
   ```

   `tensorzero.toml` uses async Postgres writes (pool size 50). For batched inserts, use `tensorzero.postgres.batch.toml`. To isolate gateway CPU, use `tensorzero.no-obs.toml` and unset `TENSORZERO_POSTGRES_URL`.

3. Install [vegeta](https://github.com/tsenart/vegeta) (`brew install vegeta`).

## Run

```bash
cd crates/tensorzero-core/tests/load/synapse-compat
./run.sh
```

Optional env:

- `GATEWAY_URL` (default `http://127.0.0.1:3000`)
- `DURATION` (default `8s`)
- `TIMEOUT` (default `5s`)
- `WORKERS` (default `8,32,64,128`)
- `RESULTS_DIR` (default `/tmp/synapse-compat-load`)
- `SCENARIOS` (comma-separated names; empty means all)
- `SLEEP_BETWEEN` (seconds between worker levels, default `2`)

`run.sh` uses vegeta `-rate=0` (as-fast-as-possible) with `-max-workers=N` so the knee is concurrency, not a fixed RPS target. Dummy sleeps 1ms per non-streaming inference; streaming emits 16 chunks.

Prefer `tensorzero.no-obs.toml` for endpoint-vs-endpoint comparison. Observability-on sweeps should use Postgres, not ClickHouse.

Long-reasoning chat (40k thinking tokens + 10k text tokens at 100 tok/s, ~500s per request) is opt-in and samples gateway RSS (KB) while vegeta runs:

```bash
SCENARIOS=chat-long-stream WORKERS=1,8,32 DURATION=15s TIMEOUT=600s ./run.sh
```

Scenarios: chat (cache off / cache on), streaming, stream-aggregate, Anthropic `/v1/messages`, embeddings, rerank, completions, responses, `/internal/synapse/balances`.

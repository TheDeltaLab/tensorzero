#!/usr/bin/env bash
# Modified by Delta-AI under Apache 2.0
# Closed-loop concurrency sweep for Synapse-compatible endpoints (dummy LLM).
#
# Usage:
#   GATEWAY_URL=http://127.0.0.1:3000 ./run.sh
#
# Requires vegeta. Gateway must be started with `--features e2e_tests`
# so `dummy::` is available.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GATEWAY_URL="${GATEWAY_URL:-http://127.0.0.1:3000}"
DURATION="${DURATION:-8s}"
TIMEOUT="${TIMEOUT:-5s}"
WORKERS="${WORKERS:-8,32,64,128}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/synapse-compat-load}"
# Comma-separated scenario names; empty means all.
SCENARIOS="${SCENARIOS:-}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-2}"
mkdir -p "$RESULTS_DIR"

if ! command -v vegeta >/dev/null 2>&1; then
  echo "vegeta is required (brew install vegeta)" >&2
  exit 1
fi

if ! curl -fsS "$GATEWAY_URL/status" >/dev/null; then
  echo "gateway not reachable at $GATEWAY_URL/status" >&2
  exit 1
fi

IFS=',' read -r -a WORKER_LIST <<<"$WORKERS"

run_one() {
  local name="$1"
  local method="$2"
  local path="$3"
  local body="$4"
  shift 4
  # Bash 3.2 + `set -u`: `arr=("$@")` leaves the name unset when there are no args.
  local extra_headers=()
  if [ "$#" -gt 0 ]; then
    extra_headers=("$@")
  fi

  local workers
  for workers in "${WORKER_LIST[@]}"; do
    local out="$RESULTS_DIR/${name}.w${workers}.bin"
    local json="$RESULTS_DIR/${name}.w${workers}.json"
    local hdrs=(-header "Content-Type: application/json")
    local h
    for h in "${extra_headers[@]+"${extra_headers[@]}"}"; do
      hdrs+=(-header "$h")
    done

    echo "==> $name workers=$workers $method $path"
    local gpid rss_file sampler
    gpid=$(lsof -nP -iTCP:3000 -sTCP:LISTEN -t 2>/dev/null | head -1 || true)
    rss_file="$RESULTS_DIR/${name}.w${workers}.rss.csv"
    echo "ts_unix,rss_kb" >"$rss_file"
    if [ -n "${gpid:-}" ]; then
      (
        while kill -0 "$gpid" 2>/dev/null; do
          rss=$(ps -o rss= -p "$gpid" 2>/dev/null | tr -d ' ')
          echo "$(date +%s),${rss:-}"
          sleep 1
        done
      ) >>"$rss_file" &
      sampler=$!
    else
      sampler=""
    fi
    echo "$method ${GATEWAY_URL}${path}" |
      vegeta attack \
        "${hdrs[@]}" \
        ${body:+-body="$body"} \
        -http2=false \
        -duration="$DURATION" \
        -timeout="$TIMEOUT" \
        -rate=0 \
        -max-workers="$workers" \
        -output="$out"
    if [ -n "$sampler" ]; then
      kill "$sampler" 2>/dev/null || true
      wait "$sampler" 2>/dev/null || true
    fi
    vegeta report -type=json <"$out" >"$json"
    vegeta report <"$out"
    if [ -n "${gpid:-}" ]; then
      echo "RSS kb (min/avg/max): $(python3 -c "
import csv,sys
rows=[int(r['rss_kb']) for r in csv.DictReader(open('$rss_file')) if r.get('rss_kb','').isdigit()]
print('n/a' if not rows else f'{min(rows)}/{sum(rows)//len(rows)}/{max(rows)}')
")"
    fi
    echo
    sleep "$SLEEP_BETWEEN"
  done
}

should_run() {
  local name="$1"
  if [ -z "$SCENARIOS" ]; then
    return 0
  fi
  case ",$SCENARIOS," in
    *",$name,"*) return 0 ;;
    *) return 1 ;;
  esac
}

BODIES="$SCRIPT_DIR/bodies"

if should_run chat-cache-off; then
  run_one chat-cache-off POST /v1/chat/completions "$BODIES/chat.json" "x-synapse-cache: false"
fi
if should_run chat-cache-on; then
  run_one chat-cache-on POST /v1/chat/completions "$BODIES/chat.json"
fi
if should_run chat-stream; then
  run_one chat-stream POST /v1/chat/completions "$BODIES/chat-stream.json" "x-synapse-cache: false"
fi
if should_run chat-stream-agg; then
  run_one chat-stream-agg POST /v1/chat/completions "$BODIES/chat-stream.json" \
    "x-synapse-cache: false" \
    'x-synapse-stream-aggregate: [{"part":"content","startDelayMs":0,"intervalMs":10000,"maxChars":5000}]'
fi
if should_run messages; then
  run_one messages POST /v1/messages "$BODIES/messages.json" "x-synapse-cache: false"
fi
if should_run embeddings; then
  run_one embeddings POST /v1/embeddings "$BODIES/embeddings.json" "x-synapse-cache: false"
fi
if should_run rerank; then
  run_one rerank POST /v1/rerank "$BODIES/rerank.json" "x-synapse-provider: dummy"
fi
if should_run completions; then
  run_one completions POST /v1/completions "$BODIES/completions.json" "x-synapse-cache: false"
fi
if should_run responses; then
  run_one responses POST /v1/responses "$BODIES/responses.json" "x-synapse-cache: false"
fi
if should_run balances; then
  run_one balances GET /internal/synapse/balances ""
fi
# 40k thinking + 10k text at 100 tok/s (~500s/request). Opt-in only.
if [ -n "$SCENARIOS" ] && should_run chat-long-stream; then
  run_one chat-long-stream POST /v1/chat/completions "$BODIES/chat-long-stream.json" "x-synapse-cache: false"
fi

python3 "$SCRIPT_DIR/summarize.py" "$RESULTS_DIR"

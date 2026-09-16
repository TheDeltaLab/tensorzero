# TensorZero TypeScript SDK

TypeScript clients for the TensorZero gateway's **async inference API**:

| Package | Purpose |
| --- | --- |
| `@delta-ai/tensorzero-sdk` | Zero-dependency client for the async job HTTP API (submit / poll / stream / wait) |
| `@delta-ai/ai-sdk-provider` | Vercel AI SDK provider (`createTensorZero`) with batch support on top of the async API |

Requires Node.js 18+ or a modern browser (global `fetch`, `ReadableStream`). No runtime dependencies.

## Installation

Both packages are published to the public npm registry:

```bash
pnpm add @delta-ai/tensorzero-sdk
# and, for the AI SDK integration:
pnpm add @delta-ai/ai-sdk-provider ai
```

## `@delta-ai/tensorzero-sdk`

```ts
import { createTensorZeroClient } from "@delta-ai/tensorzero-sdk";

const client = createTensorZeroClient({
  baseURL: "https://gateway.example.com", // gateway origin; `/v1/...` paths are appended
  apiKey: process.env.TENSORZERO_API_KEY,  // sent as `Authorization: Bearer ...`
  // fetch: myFetch,                       // optional custom fetch
  // headers: { "x-request-id": "..." },   // optional extra headers
});
```

### Submit

`submit` enqueues a durable task and returns `202` with a task id. The body is
the raw JSON of the target API (`stream` is forced on internally by the
gateway; execution is always streaming so events can be relayed).

```ts
const { taskId } = await client.submitChatCompletion({
  model: "openai::gpt-5",
  messages: [{ role: "user", content: "Write a haiku about queues" }],
});

// Equivalent generic form; also: submitResponses / submitMessages
await client.submit("chat", { model: "openai::gpt-5", messages: [/* ... */] });
await client.submitResponses({ model: "openai::gpt-5", input: "..." });
await client.submitMessages({ model: "anthropic::claude-sonnet-4-5", max_tokens: 1024, messages: [/* ... */] });
```

### Poll

```ts
const status = await client.getTask(taskId);
// Discriminated union on `status`:
//   { status: "queued",    taskId, queuePosition? }
//   { status: "running",   taskId, startedAt?, elapsedMs? }
//   { status: "completed", taskId, response }   // full sync-API-shaped response body
//   { status: "failed",    taskId, error? }
//   / { status: "cancelled", taskId, error? }
```

### Wait (exponential backoff)

```ts
const final = await client.waitForCompletion(taskId, {
  intervalMs: 1_000,     // initial poll interval (default 1s)
  maxIntervalMs: 10_000, // backoff cap (default 10s)
  timeoutMs: 300_000,    // throws TensorZeroTimeoutError after this (default: none)
});
if (final.status === "completed") {
  console.log(final.response); // e.g. a chat completion body
}
```

### Stream

`streamTask` attaches to the task's SSE event stream and is **wire-faithful**:
it yields exactly the frames the gateway sends — incremental chunks (for chat
tasks, sync-streaming-shaped chunk events ending with `data: [DONE]`) and, on
failure, the terminal `event: error` frame. Nothing is synthesized: a
successful stream ends with a bare EOF, so the iterable simply ends.

The SDK handles transport concerns transparently: the server replays
already-produced events on attach, so the SDK deduplicates them by sequence
number and reconnects automatically on network failures; if the event stream
is already gone (HTTP 410, e.g. expired TTL), it falls back to polling until
the task reaches a terminal state and then ends.

```ts
for await (const event of client.streamTask(taskId, { signal: abortSignal })) {
  // event.sequence: monotonic, deduplicated across reconnects
  // event.event:    SSE event name (undefined for default events, "error" for the terminal error frame)
  // event.data:     raw payload; event.json: parsed JSON when applicable
  if (event.event === "error") console.error("task failed:", event.json);
  else console.log(event.data);
}
```

The final result is **not** part of the stream. The division of labor:
`streamTask` only relays live wire events (incremental chunks + a possible
error frame); the final, complete response always comes from
`getTask` / `waitForCompletion`:

```ts
const final = await client.waitForCompletion(taskId);
if (final.status === "completed") console.log(final.response);
```

### Health

```ts
await client.status(); // GET /status (no auth)
await client.health(); // GET /health (no auth)
```

### Errors

All errors extend `TensorZeroError`:

| Class | Condition |
| --- | --- |
| `TaskNotFoundError` | 404 — unknown task id |
| `StreamGoneError` | 410 — event stream expired/gone (handled internally by `streamTask`) |
| `AsyncInferenceDisabledError` | 500 — gateway lacks async inference config |
| `TensorZeroHttpError` | any other non-2xx |
| `TensorZeroTimeoutError` | `waitForCompletion` exceeded `timeoutMs` |
| `TensorZeroStreamError` | event stream failed past the reconnect budget |
| `TensorZeroParseError` | response didn't match the expected wire shape |

## `@delta-ai/ai-sdk-provider`

```ts
import { createTensorZero } from "@delta-ai/ai-sdk-provider";
import { generateText, streamText } from "ai";

const tensorzero = createTensorZero({
  baseURL: "https://gateway.example.com/v1", // OpenAI-compatible base URL, incl. /v1
  apiKey: process.env.TENSORZERO_API_KEY,
});

// Synchronous generation/streaming go through the gateway's OpenAI-compatible API.
const { text } = await generateText({
  model: tensorzero("openai::gpt-5"),
  prompt: "Hello",
});

const stream = await streamText({
  model: tensorzero("openai::gpt-5"),
  prompt: "Tell me a story",
});
for await (const delta of stream.textStream) process.stdout.write(delta);
```

Gateway-private parameters pass through `providerOptions.tensorzero` into the
request body:

```ts
await generateText({
  model: tensorzero("openai::gpt-5"),
  prompt: "Hello",
  providerOptions: { tensorzero: { cache_options: { enabled: true } } },
});
```

### Batch (async tasks)

The chat model structurally implements `BatchModelV4`
(`Experimental_BatchLanguageModelV4` from `@ai-sdk/provider`). Starting a batch
submits each request as an independent durable async task; the returned batch
reference is a serializable JSON string encoding the `(request id, task_id)`
pairs.

With `ai` v7:

```ts
import {
  experimental_startTextBatch,
  experimental_getBatchStatus,
  experimental_getBatchResults,
} from "ai";

const model = tensorzero("openai::gpt-5");

const batch = await experimental_startTextBatch({
  model,
  requests: [
    { id: "req-1", prompt: "Summarize document A" },
    { id: "req-2", prompt: "Summarize document B" },
  ],
});
// `batch` is serializable — persist it and resume polling from another process.

const status = await experimental_getBatchStatus({ model, batch });
// pending | completed | failed (queued/running tasks → pending;
// any failed/cancelled terminal task → failed)

if (status.status !== "pending") {
  for await (const item of experimental_getBatchResults({ model, batch })) {
    if (item.status === "succeeded") console.log(item.id, item.text);
    else console.error(item.id, item.status, item.error);
  }
}
```

The three `experimental_do*` methods are also available directly on the model
(`experimental_doStartBatch` / `experimental_doGetBatchStatus` /
`experimental_doGetBatchResults`, per the `BatchModelV4` interface) for
`ai` versions without the high-level batch helpers.

Notes:

- `webhookUrl` is not supported by the gateway; passing one yields an
  `unsupported` warning.
- `experimental_doGetBatchResults` only converts chat-completion responses;
  submit `responses`/`messages`-style jobs via `provider.asyncClient`
  (`createTensorZeroClient`) instead.
- `provider.asyncClient` exposes the underlying `@delta-ai/tensorzero-sdk`
  client for direct submit/poll/stream/wait.

## Development

```bash
pnpm install
pnpm build      # tsup: ESM + CJS + .d.ts for both packages
pnpm typecheck  # tsc --noEmit
pnpm test       # vitest
```

The provider typechecks against the built `dist/` of `@delta-ai/tensorzero-sdk`,
so run `pnpm build` before `pnpm typecheck` on a fresh checkout.

# TensorZero Go SDK

A zero-dependency (standard library only) Go client for the TensorZero
gateway's **asynchronous inference job API**, plus its `/status` and `/health`
endpoints.

## Install

The SDK is a Go submodule of the public `TheDeltaLab/tensorzero` repository,
released as subdirectory tags of the form `sdk/go/vX.Y.Z`. Reference the plain
semver when fetching (the proxy protocol rejects the full subdirectory path as
a non-canonical version):

```sh
go get github.com/TheDeltaLab/tensorzero/sdk/go@v0.1.0
# or track the latest release:
go get github.com/TheDeltaLab/tensorzero/sdk/go@latest
```

## Quick start

```go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	tensorzero "github.com/TheDeltaLab/tensorzero/sdk/go"
)

func main() {
	client := tensorzero.NewClient("http://localhost:3000",
		tensorzero.WithAPIKey("sk-..."),
	)

	ctx := context.Background()

	// Submit: body is the raw JSON request of the corresponding sync API.
	body := json.RawMessage(`{
		"model": "openai::gpt-5",
		"messages": [{"role": "user", "content": "Hello!"}]
	}`)
	taskID, err := client.SubmitChatCompletion(ctx, body)
	// client.SubmitResponses / client.SubmitMessages also exist.
	if err != nil {
		log.Fatal(err)
	}

	// Option A: poll with backoff until the task is terminal.
	status, err := client.WaitForCompletion(ctx, taskID,
		tensorzero.WithInitialInterval(500*time.Millisecond),
		tensorzero.WithMaxInterval(5*time.Second),
		tensorzero.WithTimeout(10*time.Minute),
	)
	if err != nil {
		log.Fatal(err)
	}
	if status.Status == tensorzero.TaskCompleted {
		fmt.Println(string(status.Response)) // full sync-shaped response
	}

	// Option B: follow the live SSE event stream (incremental chunks plus, on
	// failure, a terminal `error` frame). StreamTask forwards exactly what the
	// gateway sends and never synthesizes events; the final result is not on
	// the stream — fetch it with GetTask / WaitForCompletion.
	events, errs := client.StreamTask(ctx, taskID)
	for ev := range events {
		fmt.Printf("event=%s data=%s\n", ev.Event, ev.Data)
	}
	if err := <-errs; err != nil {
		log.Fatal(err)
	}
	final, err := client.GetTask(ctx, taskID) // final TaskStatus + Response
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(final.Status, string(final.Response))
}
```

## API overview

| Method | Endpoint |
| --- | --- |
| `SubmitChatCompletion(ctx, body)` | `POST /v1/chat/completions/async` |
| `SubmitResponses(ctx, body)` | `POST /v1/responses/async` |
| `SubmitMessages(ctx, body)` | `POST /v1/messages/async` |
| `GetTask(ctx, taskID)` | `GET /v1/async_tasks/{task_id}` |
| `StreamTask(ctx, taskID)` | `GET /v1/async_tasks/{task_id}/stream` |
| `WaitForCompletion(ctx, taskID, opts...)` | polls `GetTask` |
| `Status(ctx)` / `Health(ctx)` | `GET /status`, `GET /health` (no auth) |

- `TaskStatus` is discriminated by `Status` (`TaskQueued`, `TaskRunning`,
  `TaskCompleted`, `TaskFailed`, `TaskCancelled`); `Response` is a
  `json.RawMessage` in the shape of the API the task was submitted to.
- `StreamTask` is wire-faithful: it forwards exactly the frames the gateway
  sends (incremental chunks, the OpenAI-style `data: [DONE]` sentinel when
  present, and, on failure, a terminal `event: error` frame) and never
  synthesizes events. A clean close of the events channel with no error means
  the task reached a terminal state. It reconnects on dropped connections
  (deduplicating replayed events) and, on `410 Gone` (expired stream cache),
  silently falls back to polling the status endpoint until the task is
  terminal, then ends the stream. **The final result is never on the
  stream** — always fetch it with `GetTask` or `WaitForCompletion`.
- Errors: `*TaskNotFoundError` (404), `*HTTPError` (status code + body) for
  everything else, including 500 when async inference is disabled.

## Development

```sh
cd sdk/go
go build ./...
go vet ./...
go test ./...
```

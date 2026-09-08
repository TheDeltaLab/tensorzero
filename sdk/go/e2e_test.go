// Modified by Delta-AI under Apache 2.0
package tensorzero

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
	"time"
)

// End-to-end tests against a real gateway. They run only when both
// TZ_E2E_KEY and TZ_E2E_GATEWAY are set (and are skipped under -short). The
// API key is read from the environment only — never commit it.
//
//	TZ_E2E_GATEWAY=https://gateway.example.com TZ_E2E_KEY=sk-... go test -run E2E -v
func e2eClient(t *testing.T) *Client {
	t.Helper()
	if testing.Short() {
		t.Skip("skipping e2e test in -short mode")
	}
	key := os.Getenv("TZ_E2E_KEY")
	gateway := os.Getenv("TZ_E2E_GATEWAY")
	if key == "" || gateway == "" {
		t.Skip("TZ_E2E_KEY and TZ_E2E_GATEWAY must both be set; skipping e2e test")
	}
	return NewClient(gateway, WithAPIKey(key))
}

func e2eContext(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	t.Cleanup(cancel)
	return ctx
}

var e2eChatBody = json.RawMessage(`{
	"model": "deepseek-v4-flash",
	"messages": [{"role": "user", "content": "Reply with the single word: pong"}],
	"max_tokens": 64
}`)

// chatCompletion mirrors the fields of a chat.completion response/chunk that
// these tests assert on.
type chatCompletion struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Choices []struct {
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		Delta struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"delta"`
	} `json:"choices"`
}

func TestE2EStatusAndHealth(t *testing.T) {
	client := e2eClient(t)
	ctx := e2eContext(t)

	status, err := client.Status(ctx)
	if err != nil {
		t.Fatalf("Status: %v", err)
	}
	if !json.Valid(status) || !strings.Contains(string(status), "ok") {
		t.Fatalf("status body = %s", status)
	}
	health, err := client.Health(ctx)
	if err != nil {
		t.Fatalf("Health: %v", err)
	}
	t.Logf("status=%s health=%s", status, health)
}

func TestE2ESubmitWaitCompleted(t *testing.T) {
	client := e2eClient(t)
	ctx := e2eContext(t)

	taskID, err := client.SubmitChatCompletion(ctx, e2eChatBody)
	if err != nil {
		t.Fatalf("submit: %v", err)
	}
	t.Logf("task_id=%s", taskID)

	// The first read may catch an intermediate state; it must be a valid one.
	first, err := client.GetTask(ctx, taskID)
	if err != nil {
		t.Fatalf("GetTask: %v", err)
	}
	switch first.Status {
	case TaskQueued, TaskRunning, TaskCompleted:
		t.Logf("intermediate status=%s", first.Status)
	default:
		t.Fatalf("unexpected intermediate status %q", first.Status)
	}

	final, err := client.WaitForCompletion(ctx, taskID,
		WithInitialInterval(500*time.Millisecond),
		WithMaxInterval(3*time.Second),
		WithTimeout(2*time.Minute),
	)
	if err != nil {
		t.Fatalf("WaitForCompletion: %v", err)
	}
	if final.Status != TaskCompleted {
		t.Fatalf("status = %s, error = %s", final.Status, final.Error)
	}

	var resp chatCompletion
	if err := json.Unmarshal(final.Response, &resp); err != nil {
		t.Fatalf("response is not a chat.completion: %v\n%s", err, final.Response)
	}
	if resp.Object != "chat.completion" {
		t.Fatalf("object = %q, want chat.completion", resp.Object)
	}
	if len(resp.Choices) == 0 || strings.TrimSpace(resp.Choices[0].Message.Content) == "" {
		t.Fatalf("empty choices/message content in %s", final.Response)
	}
	t.Logf("content=%q", resp.Choices[0].Message.Content)
}

func TestE2EStreamTaskWireFaithful(t *testing.T) {
	client := e2eClient(t)
	ctx := e2eContext(t)

	// A longer-running completion leaves room for the stream to follow live
	// even across transient gateway retries of the stream endpoint.
	body := json.RawMessage(`{
		"model": "deepseek-v4-flash",
		"messages": [{"role": "user", "content": "Count from 1 to 30, one number per line."}],
		"max_tokens": 400
	}`)
	taskID, err := client.SubmitChatCompletion(ctx, body)
	if err != nil {
		t.Fatalf("submit: %v", err)
	}

	events, errs := client.StreamTask(ctx, taskID)
	var chunks, doneSentinels int
	for ev := range events {
		// Wire-faithful: the gateway never emits events with these names;
		// any of them would mean the SDK synthesized a terminal event.
		switch ev.Event {
		case "completed", "failed", "cancelled":
			t.Fatalf("synthetic terminal event leaked into stream: %+v", ev)
		}
		// The gateway forwards OpenAI's `data: [DONE]` sentinel verbatim.
		if string(ev.Data) == "[DONE]" {
			doneSentinels++
			continue
		}
		var chunk chatCompletion
		if err := json.Unmarshal(ev.Data, &chunk); err != nil {
			t.Fatalf("event %q data is not a chat.completion.chunk: %v\n%s", ev.Event, err, ev.Data)
		}
		if chunk.Object != "chat.completion.chunk" {
			t.Fatalf("object = %q, want chat.completion.chunk", chunk.Object)
		}
		chunks++
	}
	if err := <-errs; err != nil {
		t.Fatalf("stream error: %v", err)
	}
	if chunks == 0 {
		t.Fatal("expected at least one chunk event")
	}
	t.Logf("received %d chunk events, %d [DONE] sentinels; stream ended quietly", chunks, doneSentinels)

	// The final result comes from GetTask, not the stream.
	final, err := client.GetTask(ctx, taskID)
	if err != nil {
		t.Fatalf("GetTask: %v", err)
	}
	if final.Status != TaskCompleted {
		t.Fatalf("status = %s after stream end", final.Status)
	}
}

func TestE2ETaskNotFound(t *testing.T) {
	client := e2eClient(t)
	_, err := client.GetTask(e2eContext(t), "00000000-0000-0000-0000-000000000000")
	var notFound *TaskNotFoundError
	if !errors.As(err, &notFound) {
		t.Fatalf("err = %v, want TaskNotFoundError", err)
	}
}

func TestE2EBadModelFails(t *testing.T) {
	client := e2eClient(t)
	ctx := e2eContext(t)

	body := json.RawMessage(`{
		"model": "definitely-missing-model",
		"messages": [{"role": "user", "content": "hi"}]
	}`)
	taskID, err := client.SubmitChatCompletion(ctx, body)
	if err != nil {
		t.Fatalf("submit: %v", err)
	}
	final, err := client.WaitForCompletion(ctx, taskID, WithTimeout(2*time.Minute))
	if err != nil {
		t.Fatalf("WaitForCompletion: %v", err)
	}
	if final.Status != TaskFailed {
		t.Fatalf("status = %s, want failed", final.Status)
	}
	if !strings.Contains(string(final.Error), "not found in model table") {
		t.Fatalf("error = %s, want it to mention the model table", final.Error)
	}
}

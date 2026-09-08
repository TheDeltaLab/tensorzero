// Modified by Delta-AI under Apache 2.0
package tensorzero

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// TaskState is the lifecycle state of an async inference task, taken from the
// `status` discriminator field of GET /v1/async_tasks/{task_id}.
type TaskState string

const (
	TaskQueued    TaskState = "queued"
	TaskRunning   TaskState = "running"
	TaskCompleted TaskState = "completed"
	TaskFailed    TaskState = "failed"
	TaskCancelled TaskState = "cancelled"
)

// TaskStatus is the decoded response of GET /v1/async_tasks/{task_id}. Which
// fields are populated depends on Status:
//
//   - TaskQueued: QueuePosition (number of claimable tasks ahead, if known).
//   - TaskRunning: StartedAt (RFC 3339), ElapsedMs.
//   - TaskCompleted: Response, the full synchronous response in the shape of
//     the API the task was submitted to (chat completions / responses /
//     messages).
//   - TaskFailed, TaskCancelled: Error.
type TaskStatus struct {
	TaskID        string          `json:"task_id"`
	Status        TaskState       `json:"status"`
	QueuePosition *int64          `json:"queue_position,omitempty"`
	StartedAt     *time.Time      `json:"started_at,omitempty"`
	ElapsedMs     *uint64         `json:"elapsed_ms,omitempty"`
	Response      json.RawMessage `json:"response,omitempty"`
	Error         json.RawMessage `json:"error,omitempty"`
}

// Terminal reports whether the task has reached a final state.
func (s TaskStatus) Terminal() bool {
	switch s.Status {
	case TaskCompleted, TaskFailed, TaskCancelled:
		return true
	}
	return false
}

// SubmitChatCompletion enqueues a chat-completions-shaped request
// (POST /v1/chat/completions/async) and returns the new task ID. body is the
// raw JSON request body of the synchronous API; a `stream` field is ignored
// by the gateway.
func (c *Client) SubmitChatCompletion(ctx context.Context, body json.RawMessage) (string, error) {
	return c.submit(ctx, "/v1/chat/completions/async", body)
}

// SubmitResponses enqueues a responses-shaped request
// (POST /v1/responses/async) and returns the new task ID.
func (c *Client) SubmitResponses(ctx context.Context, body json.RawMessage) (string, error) {
	return c.submit(ctx, "/v1/responses/async", body)
}

// SubmitMessages enqueues an Anthropic-messages-shaped request
// (POST /v1/messages/async) and returns the new task ID.
func (c *Client) SubmitMessages(ctx context.Context, body json.RawMessage) (string, error) {
	return c.submit(ctx, "/v1/messages/async", body)
}

func (c *Client) submit(ctx context.Context, path string, body json.RawMessage) (string, error) {
	if len(body) == 0 {
		return "", fmt.Errorf("tensorzero: submit %s: empty request body", path)
	}
	req, err := c.newRequest(ctx, http.MethodPost, path, bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	resp, err := c.do(req)
	if err != nil {
		return "", err
	}
	var launch struct {
		TaskID string `json:"task_id"`
	}
	if err := decodeJSON(resp, &launch); err != nil {
		return "", fmt.Errorf("tensorzero: decoding submit response: %w", err)
	}
	if launch.TaskID == "" {
		return "", fmt.Errorf("tensorzero: submit %s: response missing task_id", path)
	}
	return launch.TaskID, nil
}

// GetTask fetches the current status of an async task. It returns a
// *TaskNotFoundError if the task does not exist.
func (c *Client) GetTask(ctx context.Context, taskID string) (*TaskStatus, error) {
	req, err := c.newRequest(ctx, http.MethodGet, "/v1/async_tasks/"+taskID, nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.do(req)
	if err != nil {
		return nil, err
	}
	var status TaskStatus
	if err := decodeJSON(resp, &status); err != nil {
		return nil, fmt.Errorf("tensorzero: decoding task status: %w", err)
	}
	switch status.Status {
	case TaskQueued, TaskRunning, TaskCompleted, TaskFailed, TaskCancelled:
	default:
		return nil, fmt.Errorf("tensorzero: unknown task status %q", status.Status)
	}
	return &status, nil
}

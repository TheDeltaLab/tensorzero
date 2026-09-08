// Modified by Delta-AI under Apache 2.0
package tensorzero

import "fmt"

// HTTPError is returned for any non-success HTTP response that is not mapped
// to a more specific error type. It carries the response status code and body.
type HTTPError struct {
	StatusCode int
	Body       []byte
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("tensorzero: HTTP %d: %s", e.StatusCode, string(e.Body))
}

// TaskNotFoundError is returned when the gateway answers 404 for a task ID.
type TaskNotFoundError struct {
	TaskID string
}

func (e *TaskNotFoundError) Error() string {
	return fmt.Sprintf("tensorzero: async task %q not found", e.TaskID)
}

// StreamExpiredError indicates the gateway answered 410 Gone for a task's
// event stream (the Redis stream expired or was never written because the
// worker crashed). StreamTask handles this internally by falling back to
// polling the task status endpoint; it is never returned as a terminal error
// from the public streaming API.
type StreamExpiredError struct {
	TaskID string
}

func (e *StreamExpiredError) Error() string {
	return fmt.Sprintf("tensorzero: event stream for async task %q is no longer available", e.TaskID)
}

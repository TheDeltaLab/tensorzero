// Modified by Delta-AI under Apache 2.0
package tensorzero

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"
)

const (
	// streamReconnectInitial is the delay before the first reconnect after a
	// dropped SSE connection; it doubles up to streamReconnectMax.
	streamReconnectInitial = 200 * time.Millisecond
	streamReconnectMax     = 5 * time.Second
	// maxStreamReconnects bounds consecutive failed/interrupted connections
	// before StreamTask gives up.
	maxStreamReconnects = 5
)

// StreamTask follows a task's SSE event stream
// (GET /v1/async_tasks/{task_id}/stream), replaying buffered events and then
// live ones. It returns a channel of events and a channel that receives at
// most one terminal error; both are closed when streaming ends.
//
// StreamTask is wire-faithful: it forwards exactly the frames the gateway
// sends — incremental chunks, the OpenAI-style `data: [DONE]` sentinel when
// the underlying stream includes one, and, on failure, a terminal
// `event: error` frame — and never synthesizes events. On success the gateway
// closes the stream with no terminal marker, so a clean end of the events
// channel (with no error on the error channel) is the success signal. The
// final result is never on the stream: fetch it with GetTask or
// WaitForCompletion.
//
// The stream is resilient:
//
//   - If the connection drops before the task reaches a terminal state,
//     StreamTask reconnects. The gateway replays the stream from the
//     beginning, so already-delivered events are skipped (deduplicated).
//   - If the gateway answers 410 Gone (the stream cache expired, or the
//     worker crashed before writing any event), StreamTask silently falls
//     back to polling the status endpoint until the task is terminal, then
//     ends the stream quietly.
//
// End-of-stream handling: when the SSE connection closes without a terminal
// error frame, StreamTask checks the status endpoint once to decide between
// ending quietly (task terminal) and reconnecting (connection dropped); that
// status is only used for the decision and is not forwarded.
//
// Cancelling ctx stops the stream; the error channel then receives ctx.Err().
func (c *Client) StreamTask(ctx context.Context, taskID string) (<-chan TaskEvent, <-chan error) {
	events := make(chan TaskEvent)
	errs := make(chan error, 1)
	go func() {
		defer close(events)
		defer close(errs)
		if err := c.runStream(ctx, taskID, events); err != nil {
			errs <- err
		}
	}()
	return events, errs
}

func (c *Client) runStream(ctx context.Context, taskID string, events chan<- TaskEvent) error {
	delivered := 0
	failures := 0
	delay := streamReconnectInitial
	for {
		n, err := c.streamOnce(ctx, taskID, delivered, events)
		delivered += n

		var expired *StreamExpiredError
		switch {
		case errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded):
			return err
		case errors.As(err, &expired):
			// The stream cache expired (or the worker crashed before writing
			// the first event) and the task is terminal: poll the status
			// endpoint until the terminal state is reached, then end quietly.
			if _, werr := c.WaitForCompletion(ctx, taskID); werr != nil {
				return werr
			}
			return nil
		case errors.Is(err, io.EOF):
			// The stream closed without a terminal marker. Success markers
			// carry no payload, so the task may simply be done — or the
			// connection dropped mid-task. The status endpoint disambiguates;
			// its result drives the end-vs-reconnect decision only.
			status, serr := c.GetTask(ctx, taskID)
			if serr != nil {
				return serr
			}
			if status.Terminal() {
				return nil
			}
			failures++
		default:
			var notFound *TaskNotFoundError
			if errors.As(err, &notFound) {
				return err
			}
			failures++
		}

		if failures > maxStreamReconnects {
			if err != nil && !errors.Is(err, io.EOF) {
				return fmt.Errorf("tensorzero: stream for task %q failed after %d attempts: %w", taskID, failures, err)
			}
			return fmt.Errorf("tensorzero: stream for task %q closed %d times without reaching a terminal state", taskID, failures)
		}
		timer := time.NewTimer(delay)
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
		}
		delay *= 2
		if delay > streamReconnectMax {
			delay = streamReconnectMax
		}
	}
}

// streamOnce opens the SSE stream and forwards events, skipping the first
// skip of them (they were delivered by a previous, dropped connection). It
// returns the number of events forwarded on this connection and io.EOF when
// the stream closes cleanly.
func (c *Client) streamOnce(ctx context.Context, taskID string, skip int, events chan<- TaskEvent) (int, error) {
	req, err := c.newRequest(ctx, http.MethodGet, "/v1/async_tasks/"+taskID+"/stream", nil)
	if err != nil {
		return 0, err
	}
	req.Header.Set("Accept", "text/event-stream")
	resp, err := c.do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	scanner := newSSEScanner(resp.Body)
	delivered := 0
	for {
		ev, err := scanner.next()
		if err != nil {
			return delivered, err
		}
		if skip > 0 {
			skip--
			continue
		}
		select {
		case events <- ev:
			delivered++
		case <-ctx.Done():
			return delivered, ctx.Err()
		}
	}
}

// Modified by Delta-AI under Apache 2.0
package tensorzero

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

// sequenceServer serves the given status bodies in order, repeating the last
// one, and records the time of each request.
func sequenceServer(t *testing.T, bodies []string) (*httptest.Server, *int64, *[]time.Time) {
	t.Helper()
	var calls int64
	var times []time.Time
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		times = append(times, time.Now())
		n := atomic.AddInt64(&calls, 1)
		i := int(n) - 1
		if i >= len(bodies) {
			i = len(bodies) - 1
		}
		io.WriteString(w, bodies[i])
	}))
	t.Cleanup(srv.Close)
	return srv, &calls, &times
}

func TestWaitForCompletionPollsUntilTerminal(t *testing.T) {
	queued := `{"status":"queued","task_id":"t","queue_position":1}`
	running := `{"status":"running","task_id":"t"}`
	completed := `{"status":"completed","task_id":"t","response":{"id":"r1"}}`
	srv, calls, times := sequenceServer(t, []string{queued, queued, running, completed})

	client := NewClient(srv.URL)
	status, err := client.WaitForCompletion(context.Background(), "t",
		WithInitialInterval(10*time.Millisecond), WithMaxInterval(40*time.Millisecond))
	if err != nil {
		t.Fatalf("WaitForCompletion: %v", err)
	}
	if status.Status != TaskCompleted {
		t.Fatalf("status = %q", status.Status)
	}
	if got := atomic.LoadInt64(calls); got != 4 {
		t.Fatalf("calls = %d, want 4", got)
	}
	// Exponential backoff: 10ms, 20ms between the four polls — the second gap
	// must be roughly double the first.
	gap1 := (*times)[1].Sub((*times)[0])
	gap2 := (*times)[2].Sub((*times)[1])
	if gap2 < gap1 {
		t.Fatalf("backoff did not grow: gap1=%v gap2=%v", gap1, gap2)
	}
}

func TestWaitForCompletionImmediateTerminal(t *testing.T) {
	srv, calls, _ := sequenceServer(t, []string{`{"status":"failed","task_id":"t","error":{"message":"boom"}}`})
	status, err := NewClient(srv.URL).WaitForCompletion(context.Background(), "t")
	if err != nil {
		t.Fatalf("WaitForCompletion: %v", err)
	}
	if status.Status != TaskFailed {
		t.Fatalf("status = %q", status.Status)
	}
	if got := atomic.LoadInt64(calls); got != 1 {
		t.Fatalf("calls = %d, want 1", got)
	}
}

func TestWaitForCompletionContextCancel(t *testing.T) {
	srv, _, _ := sequenceServer(t, []string{`{"status":"queued","task_id":"t"}`})
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(30 * time.Millisecond)
		cancel()
	}()
	_, err := NewClient(srv.URL).WaitForCompletion(ctx, "t",
		WithInitialInterval(time.Hour), WithTimeout(0))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
}

func TestWaitForCompletionTimeout(t *testing.T) {
	srv, _, _ := sequenceServer(t, []string{`{"status":"running","task_id":"t"}`})
	_, err := NewClient(srv.URL).WaitForCompletion(context.Background(), "t",
		WithInitialInterval(5*time.Millisecond), WithTimeout(40*time.Millisecond))
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want context.DeadlineExceeded", err)
	}
}

func TestWaitForCompletionNotFound(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()
	_, err := NewClient(srv.URL).WaitForCompletion(context.Background(), "missing",
		WithInitialInterval(time.Millisecond))
	var notFound *TaskNotFoundError
	if !errors.As(err, &notFound) {
		t.Fatalf("err = %v, want TaskNotFoundError", err)
	}
}

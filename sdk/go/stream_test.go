// Modified by Delta-AI under Apache 2.0
package tensorzero

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

// streamServer emulates the gateway's /v1/async_tasks/{id}[/stream] pair.
type streamServer struct {
	t *testing.T

	// streamHandler is called per stream request; n is the 1-based attempt.
	streamHandler func(w http.ResponseWriter, n int)
	// statusBodies are served in order per status poll, last one repeated.
	statusBodies []string

	streamCalls int64
	statusCalls int64
}

func (s *streamServer) start() *httptest.Server {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/v1/async_tasks/"+testTaskID+"/stream":
			n := atomic.AddInt64(&s.streamCalls, 1)
			s.streamHandler(w, int(n))
		case r.URL.Path == "/v1/async_tasks/"+testTaskID:
			n := atomic.AddInt64(&s.statusCalls, 1)
			i := int(n) - 1
			if i >= len(s.statusBodies) {
				i = len(s.statusBodies) - 1
			}
			io.WriteString(w, s.statusBodies[i])
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	s.t.Cleanup(srv.Close)
	return srv
}

func writeSSE(w http.ResponseWriter, event, data string) {
	w.Header().Set("Content-Type", "text/event-stream")
	if event == "" {
		fmt.Fprintf(w, "data: %s\n\n", data)
	} else {
		fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, data)
	}
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

// collect reads events until both channels close and returns the events plus
// the terminal error, if any.
func collect(events <-chan TaskEvent, errs <-chan error) ([]TaskEvent, error) {
	var out []TaskEvent
	for ev := range events {
		out = append(out, ev)
	}
	// errs is buffered and closed by the time events closes.
	return out, <-errs
}

func TestStreamTaskEventsThenTerminal(t *testing.T) {
	srv := (&streamServer{
		t: t,
		streamHandler: func(w http.ResponseWriter, n int) {
			writeSSE(w, "chat.completion.chunk", `{"delta":{"content":"Hel"}}`)
			writeSSE(w, "chat.completion.chunk", `{"delta":{"content":"lo"}}`)
			// Success: the gateway closes the stream with no terminal marker.
		},
		statusBodies: []string{
			`{"status":"completed","task_id":"` + testTaskID + `","response":{"id":"r1"}}`,
		},
	}).start()

	events, errs := NewClient(srv.URL).StreamTask(context.Background(), testTaskID)
	got, err := collect(events, errs)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	// Exactly the frames the server sent; no synthetic terminal event.
	if len(got) != 2 {
		t.Fatalf("got %d events: %+v", len(got), got)
	}
	if got[0].Event != "chat.completion.chunk" || string(got[0].Data) != `{"delta":{"content":"Hel"}}` {
		t.Fatalf("event 0 = %+v", got[0])
	}
	if got[1].Event != "chat.completion.chunk" || string(got[1].Data) != `{"delta":{"content":"lo"}}` {
		t.Fatalf("event 1 = %+v", got[1])
	}
	// The terminal state is available via GetTask, not the stream.
	status, err := NewClient(srv.URL).GetTask(context.Background(), testTaskID)
	if err != nil || status.Status != TaskCompleted {
		t.Fatalf("GetTask = %+v, %v", status, err)
	}
}

func TestStreamTaskTerminalErrorEvent(t *testing.T) {
	srv := (&streamServer{
		t: t,
		streamHandler: func(w http.ResponseWriter, n int) {
			writeSSE(w, "chat.completion.chunk", `{"delta":{}}`)
			writeSSE(w, "error", `{"error":{"message":"boom"}}`)
		},
		statusBodies: []string{
			`{"status":"failed","task_id":"` + testTaskID + `","error":{"message":"boom"}}`,
		},
	}).start()

	got, err := collect(NewClient(srv.URL).StreamTask(context.Background(), testTaskID))
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	// The gateway's real error frame is forwarded; nothing is synthesized.
	if len(got) != 2 || got[1].Event != "error" || string(got[1].Data) != `{"error":{"message":"boom"}}` {
		t.Fatalf("events = %+v", got)
	}
}

func TestStreamTaskReconnectDeduplicates(t *testing.T) {
	srv := (&streamServer{
		t: t,
		streamHandler: func(w http.ResponseWriter, n int) {
			switch n {
			case 1:
				// First connection delivers two events, then drops mid-task.
				writeSSE(w, "chunk", `1`)
				writeSSE(w, "chunk", `2`)
			default:
				// The gateway replays from the beginning on reconnect.
				writeSSE(w, "chunk", `1`)
				writeSSE(w, "chunk", `2`)
				writeSSE(w, "chunk", `3`)
			}
		},
		statusBodies: []string{
			`{"status":"running","task_id":"` + testTaskID + `"}`,
			`{"status":"completed","task_id":"` + testTaskID + `","response":{}}`,
		},
	}).start()

	got, err := collect(NewClient(srv.URL).StreamTask(context.Background(), testTaskID))
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	want := []string{"1", "2", "3"}
	if len(got) != len(want) {
		t.Fatalf("events = %+v", got)
	}
	for i, w := range want {
		if got[i].Event != "chunk" || string(got[i].Data) != w {
			t.Fatalf("event %d = %+v, want data %s", i, got[i], w)
		}
	}
}

func TestStreamTaskGoneFallsBackToPolling(t *testing.T) {
	srv := (&streamServer{
		t: t,
		streamHandler: func(w http.ResponseWriter, n int) {
			w.WriteHeader(http.StatusGone)
			io.WriteString(w, `{"error":{"message":"stream no longer available"}}`)
		},
		statusBodies: []string{
			`{"status":"completed","task_id":"` + testTaskID + `","response":{"id":"r1"}}`,
		},
	}).start()

	got, err := collect(NewClient(srv.URL).StreamTask(context.Background(), testTaskID))
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	// 410 fallback polls until terminal and then ends quietly: the stream
	// produced nothing because the gateway sent nothing.
	if len(got) != 0 {
		t.Fatalf("events = %+v", got)
	}
	// The final result comes from GetTask / WaitForCompletion.
	status, err := NewClient(srv.URL).GetTask(context.Background(), testTaskID)
	if err != nil || status.Status != TaskCompleted {
		t.Fatalf("GetTask = %+v, %v", status, err)
	}
}

func TestStreamTaskNotFound(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()
	_, err := collect(NewClient(srv.URL).StreamTask(context.Background(), testTaskID))
	var notFound *TaskNotFoundError
	if !errors.As(err, &notFound) {
		t.Fatalf("err = %v, want TaskNotFoundError", err)
	}
}

func TestStreamTaskContextCancel(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		<-r.Context().Done() // hold the stream open until the client goes away
	}))
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(30 * time.Millisecond)
		cancel()
	}()
	_, err := collect(NewClient(srv.URL).StreamTask(ctx, testTaskID))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
}

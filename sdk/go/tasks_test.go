// Modified by Delta-AI under Apache 2.0
package tensorzero

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

const testTaskID = "0190f9c4-8e3a-7b3d-9c1e-2f4a5b6c7d8e"

func TestGetTaskParsesStatuses(t *testing.T) {
	cases := []struct {
		name     string
		body     string
		check    func(t *testing.T, s *TaskStatus)
		terminal bool
	}{
		{
			name: "queued with position",
			body: `{"status":"queued","task_id":"` + testTaskID + `","queue_position":3}`,
			check: func(t *testing.T, s *TaskStatus) {
				if s.Status != TaskQueued || s.QueuePosition == nil || *s.QueuePosition != 3 {
					t.Fatalf("got %+v", s)
				}
			},
		},
		{
			name: "queued without position",
			body: `{"status":"queued","task_id":"` + testTaskID + `"}`,
			check: func(t *testing.T, s *TaskStatus) {
				if s.Status != TaskQueued || s.QueuePosition != nil {
					t.Fatalf("got %+v", s)
				}
			},
		},
		{
			name: "running with timing",
			body: `{"status":"running","task_id":"` + testTaskID + `","started_at":"2026-09-03T10:00:00Z","elapsed_ms":1500}`,
			check: func(t *testing.T, s *TaskStatus) {
				if s.Status != TaskRunning || s.StartedAt == nil || s.ElapsedMs == nil || *s.ElapsedMs != 1500 {
					t.Fatalf("got %+v", s)
				}
				if s.StartedAt.Format("2006-01-02T15:04:05Z") != "2026-09-03T10:00:00Z" {
					t.Fatalf("bad started_at: %v", s.StartedAt)
				}
			},
		},
		{
			name: "running without timing",
			body: `{"status":"running","task_id":"` + testTaskID + `"}`,
			check: func(t *testing.T, s *TaskStatus) {
				if s.Status != TaskRunning || s.StartedAt != nil || s.ElapsedMs != nil {
					t.Fatalf("got %+v", s)
				}
			},
		},
		{
			name:     "completed carries raw response",
			body:     `{"status":"completed","task_id":"` + testTaskID + `","response":{"id":"chatcmpl-123","choices":[]}}`,
			terminal: true,
			check: func(t *testing.T, s *TaskStatus) {
				var resp struct {
					ID string `json:"id"`
				}
				if err := json.Unmarshal(s.Response, &resp); err != nil || resp.ID != "chatcmpl-123" {
					t.Fatalf("response=%s err=%v", s.Response, err)
				}
			},
		},
		{
			name:     "failed with error",
			body:     `{"status":"failed","task_id":"` + testTaskID + `","error":{"message":"boom"}}`,
			terminal: true,
			check: func(t *testing.T, s *TaskStatus) {
				if !strings.Contains(string(s.Error), "boom") {
					t.Fatalf("error=%s", s.Error)
				}
			},
		},
		{
			name:     "cancelled without error",
			body:     `{"status":"cancelled","task_id":"` + testTaskID + `"}`,
			terminal: true,
			check: func(t *testing.T, s *TaskStatus) {
				if s.Status != TaskCancelled || len(s.Error) != 0 {
					t.Fatalf("got %+v", s)
				}
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/v1/async_tasks/"+testTaskID || r.Method != http.MethodGet {
					t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
				}
				w.Header().Set("Content-Type", "application/json")
				io.WriteString(w, tc.body)
			}))
			defer srv.Close()

			status, err := NewClient(srv.URL).GetTask(context.Background(), testTaskID)
			if err != nil {
				t.Fatalf("GetTask: %v", err)
			}
			if status.TaskID != testTaskID {
				t.Fatalf("task id = %q", status.TaskID)
			}
			if status.Terminal() != tc.terminal {
				t.Fatalf("terminal = %v, want %v", status.Terminal(), tc.terminal)
			}
			tc.check(t, status)
		})
	}
}

func TestGetTaskUnknownStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"status":"mystery","task_id":"x"}`)
	}))
	defer srv.Close()
	if _, err := NewClient(srv.URL).GetTask(context.Background(), "x"); err == nil {
		t.Fatal("expected error for unknown status")
	}
}

func TestGetTaskNotFound(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		io.WriteString(w, `{"error":{"message":"no such task"}}`)
	}))
	defer srv.Close()
	_, err := NewClient(srv.URL).GetTask(context.Background(), testTaskID)
	var notFound *TaskNotFoundError
	if !errors.As(err, &notFound) {
		t.Fatalf("expected TaskNotFoundError, got %v", err)
	}
	if notFound.TaskID != testTaskID {
		t.Fatalf("task id = %q", notFound.TaskID)
	}
}

func TestGetTaskFeatureDisabled(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		io.WriteString(w, `{"error":{"message":"async inference is not enabled"}}`)
	}))
	defer srv.Close()
	_, err := NewClient(srv.URL).GetTask(context.Background(), testTaskID)
	var httpErr *HTTPError
	if !errors.As(err, &httpErr) {
		t.Fatalf("expected HTTPError, got %v", err)
	}
	if httpErr.StatusCode != 500 || !strings.Contains(string(httpErr.Body), "not enabled") {
		t.Fatalf("got %+v", httpErr)
	}
}

func TestSubmitEndpoints(t *testing.T) {
	cases := []struct {
		name   string
		submit func(*Client, context.Context, json.RawMessage) (string, error)
		path   string
	}{
		{"chat completions", (*Client).SubmitChatCompletion, "/v1/chat/completions/async"},
		{"responses", (*Client).SubmitResponses, "/v1/responses/async"},
		{"messages", (*Client).SubmitMessages, "/v1/messages/async"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var gotBody []byte
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != tc.path || r.Method != http.MethodPost {
					t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
				}
				if ct := r.Header.Get("Content-Type"); ct != "application/json" {
					t.Errorf("content-type = %q", ct)
				}
				if auth := r.Header.Get("Authorization"); auth != "Bearer test-key" {
					t.Errorf("authorization = %q", auth)
				}
				if h := r.Header.Get("X-Request-Id"); h != "req-1" {
					t.Errorf("custom header = %q", h)
				}
				gotBody, _ = io.ReadAll(r.Body)
				w.WriteHeader(http.StatusAccepted)
				io.WriteString(w, `{"task_id":"`+testTaskID+`"}`)
			}))
			defer srv.Close()

			client := NewClient(srv.URL+"/", WithAPIKey("test-key"), WithHeader("X-Request-Id", "req-1"))
			body := json.RawMessage(`{"model":"dummy::good","messages":[]}`)
			taskID, err := tc.submit(client, context.Background(), body)
			if err != nil {
				t.Fatalf("submit: %v", err)
			}
			if taskID != testTaskID {
				t.Fatalf("task id = %q", taskID)
			}
			if string(gotBody) != string(body) {
				t.Fatalf("body = %s", gotBody)
			}
		})
	}
}

func TestSubmitError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, `{"error":{"message":"invalid body"}}`)
	}))
	defer srv.Close()
	_, err := NewClient(srv.URL).SubmitChatCompletion(context.Background(), json.RawMessage(`{}`))
	var httpErr *HTTPError
	if !errors.As(err, &httpErr) || httpErr.StatusCode != 400 {
		t.Fatalf("got %v", err)
	}
}

func TestStatusAndHealth(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/status":
			io.WriteString(w, `{"status":"ok"}`)
		case "/health":
			io.WriteString(w, `{"gateway":"ok"}`)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()
	client := NewClient(srv.URL)

	status, err := client.Status(context.Background())
	if err != nil || !strings.Contains(string(status), `"ok"`) {
		t.Fatalf("status=%s err=%v", status, err)
	}
	health, err := client.Health(context.Background())
	if err != nil || !strings.Contains(string(health), `"ok"`) {
		t.Fatalf("health=%s err=%v", health, err)
	}
}

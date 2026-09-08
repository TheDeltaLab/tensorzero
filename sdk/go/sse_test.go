// Modified by Delta-AI under Apache 2.0
package tensorzero

import (
	"errors"
	"io"
	"strings"
	"testing"
)

func collectSSE(t *testing.T, input string) []TaskEvent {
	t.Helper()
	scanner := newSSEScanner(strings.NewReader(input))
	var events []TaskEvent
	for {
		ev, err := scanner.next()
		if errors.Is(err, io.EOF) {
			return events
		}
		if err != nil {
			t.Fatalf("scan: %v", err)
		}
		events = append(events, ev)
	}
}

func TestSSEScanner(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  []TaskEvent
	}{
		{
			name:  "named event with data",
			input: "event: chunk\ndata: {\"delta\":1}\n\n",
			want:  []TaskEvent{{Event: "chunk", Data: []byte(`{"delta":1}`)}},
		},
		{
			name:  "data only",
			input: "data: hello\n\n",
			want:  []TaskEvent{{Data: []byte("hello")}},
		},
		{
			name:  "comments and blank lines skipped",
			input: ": keep-alive\n\nevent: a\ndata: 1\n\n: another\n\nevent: b\ndata: 2\n\n",
			want: []TaskEvent{
				{Event: "a", Data: []byte("1")},
				{Event: "b", Data: []byte("2")},
			},
		},
		{
			name:  "multi-line data joined",
			input: "data: line1\ndata: line2\n\n",
			want:  []TaskEvent{{Data: []byte("line1\nline2")}},
		},
		{
			name:  "crlf tolerated",
			input: "event: chunk\r\ndata: {}\r\n\r\n",
			want:  []TaskEvent{{Event: "chunk", Data: []byte("{}")}},
		},
		{
			name:  "id and retry fields ignored",
			input: "id: 7\nretry: 1000\nevent: e\ndata: x\n\n",
			want:  []TaskEvent{{Event: "e", Data: []byte("x")}},
		},
		{
			name:  "no space after colon",
			input: "event:e\ndata:x\n\n",
			want:  []TaskEvent{{Event: "e", Data: []byte("x")}},
		},
		{
			name:  "pending frame dispatched at EOF",
			input: "event: error\ndata: {\"error\":{\"message\":\"boom\"}}\n",
			want:  []TaskEvent{{Event: "error", Data: []byte(`{"error":{"message":"boom"}}`)}},
		},
		{
			name:  "empty stream",
			input: "",
			want:  nil,
		},
		{
			name:  "terminal error frame after chunks",
			input: "event: chunk\ndata: 1\n\nevent: chunk\ndata: 2\n\nevent: error\ndata: {\"e\":1}\n\n",
			want: []TaskEvent{
				{Event: "chunk", Data: []byte("1")},
				{Event: "chunk", Data: []byte("2")},
				{Event: "error", Data: []byte(`{"e":1}`)},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := collectSSE(t, tc.input)
			if len(got) != len(tc.want) {
				t.Fatalf("got %d events %v, want %d", len(got), got, len(tc.want))
			}
			for i, ev := range got {
				if ev.Event != tc.want[i].Event || string(ev.Data) != string(tc.want[i].Data) {
					t.Errorf("event %d = %+v, want %+v", i, ev, tc.want[i])
				}
			}
		})
	}
}

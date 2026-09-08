// Modified by Delta-AI under Apache 2.0
package tensorzero

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"strings"
)

// TaskEvent is one event from a task's SSE stream. Event is the SSE `event:`
// name ("" if the frame carries only data) and Data is the raw JSON payload.
//
// The gateway ends the stream either by closing it after a terminal
// `event: error` frame or, on success, by closing it without any marker.
// StreamTask forwards exactly what the gateway sends and never synthesizes
// events.
type TaskEvent struct {
	Event string
	Data  json.RawMessage
}

// sseScanner parses Server-Sent Events frames from r. Comment lines
// (keep-alives starting with ':') and id/retry fields are skipped; multiple
// `data:` lines are joined with '\n' per the SSE spec.
type sseScanner struct {
	r     *bufio.Reader
	event strings.Builder
	data  bytes.Buffer
}

func newSSEScanner(r io.Reader) *sseScanner {
	return &sseScanner{r: bufio.NewReader(r)}
}

// next returns the next event, or io.EOF when the stream ends. A pending
// frame is dispatched at EOF, per the SSE spec.
func (s *sseScanner) next() (TaskEvent, error) {
	for {
		line, err := s.r.ReadString('\n')
		if len(line) > 0 {
			line = strings.TrimRight(line, "\r\n")
			s.parseLine(line)
		}
		if err != nil {
			if err == io.EOF {
				if ev, ok := s.dispatch(); ok {
					return ev, nil
				}
				return TaskEvent{}, io.EOF
			}
			return TaskEvent{}, err
		}
		if line == "" {
			if ev, ok := s.dispatch(); ok {
				return ev, nil
			}
		}
	}
}

func (s *sseScanner) parseLine(line string) {
	if line == "" || line[0] == ':' {
		return
	}
	field, value, found := strings.Cut(line, ":")
	if found {
		value = strings.TrimPrefix(value, " ")
	}
	switch field {
	case "event":
		s.event.WriteString(value)
	case "data":
		if s.data.Len() > 0 {
			s.data.WriteByte('\n')
		}
		s.data.WriteString(value)
	}
}

func (s *sseScanner) dispatch() (TaskEvent, bool) {
	if s.event.Len() == 0 && s.data.Len() == 0 {
		return TaskEvent{}, false
	}
	ev := TaskEvent{
		Event: s.event.String(),
		Data:  json.RawMessage(append([]byte(nil), s.data.Bytes()...)),
	}
	s.event.Reset()
	s.data.Reset()
	return ev, true
}

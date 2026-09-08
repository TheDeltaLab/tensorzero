// Modified by Delta-AI under Apache 2.0
// Package tensorzero is a Go client for the TensorZero gateway's asynchronous
// inference job API, plus its /status and /health endpoints. It uses only the
// Go standard library.
package tensorzero

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// maxErrorBody caps how many bytes of an error response body are kept on an
// HTTPError.
const maxErrorBody = 1 << 20

// Client calls a TensorZero gateway.
type Client struct {
	baseURL    string
	httpClient *http.Client
	headers    http.Header
}

// Option configures a Client.
type Option func(*Client)

// WithAPIKey authenticates requests with `Authorization: Bearer <key>`.
func WithAPIKey(key string) Option {
	return func(c *Client) {
		c.headers.Set("Authorization", "Bearer "+key)
	}
}

// WithHTTPClient sets the http.Client used for all requests, allowing custom
// transports, timeouts, and proxies.
func WithHTTPClient(hc *http.Client) Option {
	return func(c *Client) {
		if hc != nil {
			c.httpClient = hc
		}
	}
}

// WithHeader adds a static header sent with every request.
func WithHeader(key, value string) Option {
	return func(c *Client) {
		c.headers.Add(key, value)
	}
}

// NewClient returns a Client for the gateway at baseURL (e.g.
// "http://localhost:3000"). A trailing slash is ignored.
func NewClient(baseURL string, opts ...Option) *Client {
	c := &Client{
		baseURL:    strings.TrimSuffix(baseURL, "/"),
		httpClient: &http.Client{},
		headers:    make(http.Header),
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

func (c *Client) newRequest(ctx context.Context, method, path string, body io.Reader) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, body)
	if err != nil {
		return nil, err
	}
	for key, values := range c.headers {
		for _, value := range values {
			req.Header.Add(key, value)
		}
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	return req, nil
}

// do sends req and maps common non-success statuses to typed errors. On
// success the caller owns (and must close) resp.Body.
func (c *Client) do(req *http.Request) (*http.Response, error) {
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return resp, nil
	}
	defer resp.Body.Close()
	return nil, statusError(req.URL.Path, resp)
}

// statusError builds a typed error from a non-success response.
func statusError(path string, resp *http.Response) error {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBody))
	if resp.StatusCode == http.StatusNotFound && strings.HasPrefix(path, "/v1/async_tasks/") {
		return &TaskNotFoundError{TaskID: taskIDFromPath(path)}
	}
	if resp.StatusCode == http.StatusGone && strings.HasSuffix(path, "/stream") {
		return &StreamExpiredError{TaskID: taskIDFromPath(path)}
	}
	return &HTTPError{StatusCode: resp.StatusCode, Body: body}
}

// taskIDFromPath extracts the task ID from "/v1/async_tasks/{id}[/stream]".
func taskIDFromPath(path string) string {
	id := strings.TrimPrefix(path, "/v1/async_tasks/")
	return strings.TrimSuffix(id, "/stream")
}

func decodeJSON(resp *http.Response, v any) error {
	defer resp.Body.Close()
	return json.NewDecoder(resp.Body).Decode(v)
}

// Status calls GET {baseURL}/status and returns the raw JSON body. It does
// not require authentication.
func (c *Client) Status(ctx context.Context) (json.RawMessage, error) {
	return c.getRaw(ctx, "/status")
}

// Health calls GET {baseURL}/health and returns the raw body. It does not
// require authentication.
func (c *Client) Health(ctx context.Context) (json.RawMessage, error) {
	return c.getRaw(ctx, "/health")
}

func (c *Client) getRaw(ctx context.Context, path string) (json.RawMessage, error) {
	req, err := c.newRequest(ctx, http.MethodGet, path, nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("tensorzero: reading %s: %w", path, err)
	}
	return json.RawMessage(body), nil
}

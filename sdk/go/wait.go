// Modified by Delta-AI under Apache 2.0
package tensorzero

import (
	"context"
	"errors"
	"time"
)

const (
	defaultWaitInitialInterval = 500 * time.Millisecond
	defaultWaitMaxInterval     = 5 * time.Second
	defaultWaitTimeout         = 5 * time.Minute
)

type waitConfig struct {
	initialInterval time.Duration
	maxInterval     time.Duration
	timeout         time.Duration
}

// WaitOption configures WaitForCompletion.
type WaitOption func(*waitConfig)

// WithInitialInterval sets the delay before the first re-poll (default 500ms).
func WithInitialInterval(d time.Duration) WaitOption {
	return func(c *waitConfig) { c.initialInterval = d }
}

// WithMaxInterval caps the exponential backoff between polls (default 5s).
func WithMaxInterval(d time.Duration) WaitOption {
	return func(c *waitConfig) { c.maxInterval = d }
}

// WithTimeout bounds the total wait, independent of any deadline on the
// caller's context (default 5m). A non-positive value disables it.
func WithTimeout(d time.Duration) WaitOption {
	return func(c *waitConfig) { c.timeout = d }
}

// WaitForCompletion polls GetTask with exponential backoff (doubling from the
// initial interval up to the max interval) until the task reaches a terminal
// state. It returns the final TaskStatus, a *TaskNotFoundError if the task
// does not exist, or the context error if ctx is cancelled or the configured
// timeout elapses.
func (c *Client) WaitForCompletion(ctx context.Context, taskID string, opts ...WaitOption) (*TaskStatus, error) {
	cfg := waitConfig{
		initialInterval: defaultWaitInitialInterval,
		maxInterval:     defaultWaitMaxInterval,
		timeout:         defaultWaitTimeout,
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	if cfg.initialInterval <= 0 {
		cfg.initialInterval = defaultWaitInitialInterval
	}
	if cfg.maxInterval < cfg.initialInterval {
		cfg.maxInterval = cfg.initialInterval
	}
	if cfg.timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, cfg.timeout)
		defer cancel()
	}

	interval := cfg.initialInterval
	for {
		status, err := c.GetTask(ctx, taskID)
		if err != nil {
			var notFound *TaskNotFoundError
			if errors.As(err, &notFound) {
				return nil, err
			}
			// Other errors (e.g. feature disabled) are terminal too.
			return nil, err
		}
		if status.Terminal() {
			return status, nil
		}
		timer := time.NewTimer(interval)
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil, ctx.Err()
		case <-timer.C:
		}
		interval *= 2
		if interval > cfg.maxInterval {
			interval = cfg.maxInterval
		}
	}
}

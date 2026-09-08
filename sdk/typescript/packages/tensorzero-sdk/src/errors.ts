// Modified by Delta-AI under Apache 2.0
/** Base class for all errors thrown by the SDK. */
export class TensorZeroError extends Error {
  constructor(message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = new.target.name;
  }
}

/** A non-2xx HTTP response from the gateway. */
export class TensorZeroHttpError extends TensorZeroError {
  readonly statusCode: number;
  /** Parsed JSON error body (or raw text) returned by the gateway, if any. */
  readonly body: unknown;

  constructor(message: string, statusCode: number, body?: unknown) {
    super(message);
    this.statusCode = statusCode;
    this.body = body;
  }
}

/** 404 — the task id does not exist. */
export class TaskNotFoundError extends TensorZeroHttpError {
  constructor(message: string, body?: unknown) {
    super(message, 404, body);
  }
}

/**
 * 410 — the task has finished and its event stream is gone (expired TTL or
 * never written). Fetch the final result via `getTask` instead.
 */
export class StreamGoneError extends TensorZeroHttpError {
  constructor(message: string, body?: unknown) {
    super(message, 410, body);
  }
}

/** 500 — the gateway does not have async inference enabled. */
export class AsyncInferenceDisabledError extends TensorZeroHttpError {
  constructor(message: string, body?: unknown) {
    super(message, 500, body);
  }
}

/** `waitForCompletion` exceeded `timeoutMs` before the task reached a terminal state. */
export class TensorZeroTimeoutError extends TensorZeroError {}

/** The task event stream failed and the reconnect budget was exhausted. */
export class TensorZeroStreamError extends TensorZeroError {}

/** The gateway returned a payload that does not match the expected shape. */
export class TensorZeroParseError extends TensorZeroError {
  readonly body: unknown;

  constructor(message: string, body?: unknown) {
    super(message);
    this.body = body;
  }
}

export function isAbortError(error: unknown): boolean {
  return (
    error instanceof Error &&
    (error.name === "AbortError" || error.name === "TimeoutError")
  );
}

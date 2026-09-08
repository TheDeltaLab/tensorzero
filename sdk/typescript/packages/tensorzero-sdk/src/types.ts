// Modified by Delta-AI under Apache 2.0
/**
 * Types mirroring the TensorZero gateway async inference wire format.
 *
 * Wire reference: `AsyncTaskStatusResponse` is internally tagged on `status`
 * and serializes as `{"task_id": ..., "status": "queued", ...}` etc. The SDK
 * parses the snake_case wire fields into the camelCase shapes below.
 */

/** Which OpenAI/Anthropic-compatible API shape an async task executes. */
export type AsyncApiKind = "chat" | "responses" | "messages";

/** Body accepted by the submit endpoints: the raw JSON of the target API. */
export type AsyncRequestBody = Record<string, unknown>;

/** Response of the async submit endpoints (HTTP 202). */
export interface AsyncInferenceLaunch {
  taskId: string;
}

interface AsyncTaskBase {
  taskId: string;
}

export interface AsyncTaskQueued extends AsyncTaskBase {
  status: "queued";
  /** Number of claimable tasks ahead of this one (undefined if unknown). */
  queuePosition?: number;
}

export interface AsyncTaskRunning extends AsyncTaskBase {
  status: "running";
  /** RFC 3339 timestamp of when execution first started. */
  startedAt?: string;
  elapsedMs?: number;
}

export interface AsyncTaskCompleted extends AsyncTaskBase {
  status: "completed";
  /** Final response, in the shape of the API the task was submitted to. */
  response: unknown;
}

export interface AsyncTaskFailed extends AsyncTaskBase {
  status: "failed";
  error?: unknown;
}

export interface AsyncTaskCancelled extends AsyncTaskBase {
  status: "cancelled";
  error?: unknown;
}

/** `GET /v1/async_tasks/{task_id}` response, discriminated on `status`. */
export type AsyncTaskStatus =
  | AsyncTaskQueued
  | AsyncTaskRunning
  | AsyncTaskCompleted
  | AsyncTaskFailed
  | AsyncTaskCancelled;

export type TerminalAsyncTaskStatus =
  | AsyncTaskCompleted
  | AsyncTaskFailed
  | AsyncTaskCancelled;

export function isTerminalTaskStatus(
  status: AsyncTaskStatus,
): status is TerminalAsyncTaskStatus {
  return (
    status.status === "completed" ||
    status.status === "failed" ||
    status.status === "cancelled"
  );
}

/** One SSE event relayed from the task's event stream. */
export interface AsyncTaskStreamEvent {
  type: "event";
  /**
   * Monotonic sequence number (0-based) across reconnects. The server replays
   * the full event history on attach; the SDK deduplicates by sequence so each
   * event is yielded exactly once.
   */
  sequence: number;
  /** SSE `event:` name (`undefined` = unnamed, the SSE default). */
  event: string | undefined;
  /** Raw SSE `data:` payload. */
  data: string;
  /** `JSON.parse(data)` when the payload is valid JSON, else `undefined`. */
  json: unknown;
}

/**
 * Item yielded by `client.streamTask(...)`.
 *
 * Wire-faithful: exactly the SSE frames the gateway sends — nothing is
 * synthesized. A successful task's stream ends with a bare EOF (the `done`
 * marker is consumed server-side and never sent); a failed task's stream ends
 * after the terminal `event: error` frame. Fetch the final result via
 * `getTask` / `waitForCompletion`.
 */
export type AsyncTaskStreamItem = AsyncTaskStreamEvent;

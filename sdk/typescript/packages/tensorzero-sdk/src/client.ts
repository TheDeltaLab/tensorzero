// Modified by Delta-AI under Apache 2.0
import {
  AsyncInferenceDisabledError,
  StreamGoneError,
  TaskNotFoundError,
  TensorZeroError,
  TensorZeroHttpError,
  TensorZeroParseError,
  TensorZeroStreamError,
  TensorZeroTimeoutError,
  isAbortError,
} from "./errors.js";
import { parseSseStream } from "./sse.js";
import {
  isTerminalTaskStatus,
  type AsyncApiKind,
  type AsyncInferenceLaunch,
  type AsyncRequestBody,
  type AsyncTaskCancelled,
  type AsyncTaskFailed,
  type AsyncTaskStatus,
  type AsyncTaskStreamItem,
  type TerminalAsyncTaskStatus,
} from "./types.js";

export interface TensorZeroClientOptions {
  /**
   * Gateway origin, e.g. `https://gateway.example.com`. Endpoint paths
   * (`/v1/...`) are appended to it, so a path prefix is allowed.
   */
  baseURL: string;
  /** Bearer API key. Omit only if the gateway has auth disabled. */
  apiKey?: string;
  /** Custom fetch implementation (defaults to the global `fetch`). */
  fetch?: typeof globalThis.fetch;
  /** Extra headers sent on every request. */
  headers?: Record<string, string>;
}

export interface StreamTaskOptions {
  /** Abort the stream (in-flight HTTP request and any reconnect wait). */
  signal?: AbortSignal;
  /**
   * How many times the stream may be re-established after a network failure
   * before a `TensorZeroStreamError` is thrown. Defaults to 5.
   */
  maxReconnects?: number;
  /** Initial reconnect delay in ms (doubles each attempt, capped). Defaults to 500. */
  reconnectIntervalMs?: number;
  /** Maximum reconnect delay in ms. Defaults to 10_000. */
  maxReconnectIntervalMs?: number;
  /** Poll interval used when the event stream is gone (410 fallback). Defaults to 1_000. */
  fallbackPollIntervalMs?: number;
}

export interface WaitForCompletionOptions {
  /** Initial poll interval in ms. Defaults to 1_000. */
  intervalMs?: number;
  /** Maximum poll interval in ms (exponential backoff cap). Defaults to 10_000. */
  maxIntervalMs?: number;
  /** Give up after this many ms with a `TensorZeroTimeoutError`. Defaults to no timeout. */
  timeoutMs?: number;
  signal?: AbortSignal;
}

export interface TensorZeroClient {
  /** `POST /v1/chat/completions/async` — submit an OpenAI chat completions request. */
  submitChatCompletion(body: AsyncRequestBody): Promise<AsyncInferenceLaunch>;
  /** `POST /v1/responses/async` — submit an OpenAI responses request. */
  submitResponses(body: AsyncRequestBody): Promise<AsyncInferenceLaunch>;
  /** `POST /v1/messages/async` — submit an Anthropic messages request. */
  submitMessages(body: AsyncRequestBody): Promise<AsyncInferenceLaunch>;
  /** Generic submit for one of the three API kinds. */
  submit(kind: AsyncApiKind, body: AsyncRequestBody): Promise<AsyncInferenceLaunch>;
  /** `GET /v1/async_tasks/{taskId}` — current task status. */
  getTask(taskId: string): Promise<AsyncTaskStatus>;
  /**
   * `GET /v1/async_tasks/{taskId}/stream` as an async iterable.
   *
   * Wire-faithful: yields exactly the SSE frames the gateway sends
   * (incremental chunks, and the terminal `event: error` frame on failure) —
   * no synthetic events. Success ends with a bare EOF; the iterable simply
   * ends. Reconnects automatically on network failures and deduplicates
   * replayed events by sequence number. If the event stream is gone
   * (HTTP 410), falls back to polling the status endpoint until the task
   * reaches a terminal state, then ends.
   *
   * The final result is NOT part of the stream — fetch it with `getTask` /
   * `waitForCompletion`.
   */
  streamTask(
    taskId: string,
    options?: StreamTaskOptions,
  ): AsyncIterable<AsyncTaskStreamItem>;
  /** Poll `getTask` with exponential backoff until a terminal state. */
  waitForCompletion(
    taskId: string,
    options?: WaitForCompletionOptions,
  ): Promise<TerminalAsyncTaskStatus>;
  /** `GET /status` — gateway status (no auth required). */
  status(): Promise<unknown>;
  /** `GET /health` — gateway health check (no auth required). */
  health(): Promise<unknown>;
}

const SUBMIT_PATHS: Record<AsyncApiKind, string> = {
  chat: "/v1/chat/completions/async",
  responses: "/v1/responses/async",
  messages: "/v1/messages/async",
};

function extractMessage(body: unknown): string | undefined {
  if (
    typeof body === "object" &&
    body !== null &&
    "error" in body &&
    typeof (body as { error: unknown }).error === "object" &&
    (body as { error: unknown }).error !== null &&
    "message" in ((body as { error: object }).error as object)
  ) {
    const message = ((body as { error: { message: unknown } }).error).message;
    if (typeof message === "string") return message;
  }
  return undefined;
}

async function httpErrorFromResponse(response: Response): Promise<TensorZeroHttpError> {
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    body = undefined;
  }
  const message = extractMessage(body) ?? `HTTP ${response.status} ${response.statusText}`;
  if (response.status === 404) return new TaskNotFoundError(message, body);
  if (response.status === 410) return new StreamGoneError(message, body);
  if (response.status === 500 && /not enabled/i.test(message)) {
    return new AsyncInferenceDisabledError(message, body);
  }
  return new TensorZeroHttpError(message, response.status, body);
}

function parseAsyncTaskStatus(body: unknown): AsyncTaskStatus {
  if (typeof body !== "object" || body === null) {
    throw new TensorZeroParseError("Task status response is not an object", body);
  }
  const raw = body as Record<string, unknown>;
  if (typeof raw["task_id"] !== "string") {
    throw new TensorZeroParseError("Task status response is missing `task_id`", body);
  }
  const taskId = raw["task_id"];
  switch (raw["status"]) {
    case "queued": {
      const status: AsyncTaskStatus = { status: "queued", taskId };
      if (typeof raw["queue_position"] === "number") {
        status.queuePosition = raw["queue_position"];
      }
      return status;
    }
    case "running": {
      const status: AsyncTaskStatus = { status: "running", taskId };
      if (typeof raw["started_at"] === "string") status.startedAt = raw["started_at"];
      if (typeof raw["elapsed_ms"] === "number") status.elapsedMs = raw["elapsed_ms"];
      return status;
    }
    case "completed":
      return { status: "completed", taskId, response: raw["response"] };
    case "failed":
    case "cancelled": {
      const wire = raw["status"];
      const status: AsyncTaskFailed | AsyncTaskCancelled =
        wire === "failed"
          ? { status: "failed", taskId }
          : { status: "cancelled", taskId };
      if ("error" in raw) status.error = raw["error"];
      return status;
    }
    default:
      throw new TensorZeroParseError(
        `Unknown task status: ${JSON.stringify(raw["status"])}`,
        body,
      );
  }
}

function tryParseJson(data: string): unknown {
  try {
    return JSON.parse(data);
  } catch {
    return undefined;
  }
}

/** Sleep, rejecting immediately if the signal aborts. */
function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason ?? new DOMException("Aborted", "AbortError"));
      return;
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(signal?.reason ?? new DOMException("Aborted", "AbortError"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

export function createTensorZeroClient(
  options: TensorZeroClientOptions,
): TensorZeroClient {
  const baseURL = options.baseURL.replace(/\/+$/, "");
  const fetchImpl = options.fetch ?? globalThis.fetch;
  if (!fetchImpl) {
    throw new TensorZeroError(
      "No global `fetch` available; pass `fetch` in the client options.",
    );
  }

  const buildHeaders = (extra?: Record<string, string>): Record<string, string> => {
    const headers: Record<string, string> = { ...options.headers, ...extra };
    if (options.apiKey) {
      headers["Authorization"] = `Bearer ${options.apiKey}`;
    }
    return headers;
  };

  const requestJson = async (
    method: string,
    path: string,
    opts: { body?: unknown; signal?: AbortSignal } = {},
  ): Promise<unknown> => {
    const response = await fetchImpl(`${baseURL}${path}`, {
      method,
      headers: buildHeaders(
        opts.body !== undefined ? { "Content-Type": "application/json" } : undefined,
      ),
      body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
      signal: opts.signal ?? null,
    });
    if (!response.ok) {
      throw await httpErrorFromResponse(response);
    }
    return response.json();
  };

  const getTask = async (taskId: string): Promise<AsyncTaskStatus> => {
    const body = await requestJson("GET", `/v1/async_tasks/${encodeURIComponent(taskId)}`);
    return parseAsyncTaskStatus(body);
  };

  const waitForCompletion = async (
    taskId: string,
    waitOptions: WaitForCompletionOptions = {},
  ): Promise<TerminalAsyncTaskStatus> => {
    const intervalMs = waitOptions.intervalMs ?? 1_000;
    const maxIntervalMs = waitOptions.maxIntervalMs ?? 10_000;
    const deadline =
      waitOptions.timeoutMs !== undefined ? Date.now() + waitOptions.timeoutMs : undefined;
    let attempt = 0;
    for (;;) {
      const status = await getTask(taskId);
      if (isTerminalTaskStatus(status)) {
        return status;
      }
      const delay = Math.min(intervalMs * 2 ** attempt, maxIntervalMs);
      attempt += 1;
      if (deadline !== undefined && Date.now() + delay > deadline) {
        throw new TensorZeroTimeoutError(
          `Task ${taskId} did not reach a terminal state within ${waitOptions.timeoutMs}ms`,
        );
      }
      await sleep(delay, waitOptions.signal);
    }
  };

  async function* streamEvents(
    taskId: string,
    streamOptions: StreamTaskOptions,
  ): AsyncGenerator<AsyncTaskStreamItem> {
    const maxReconnects = streamOptions.maxReconnects ?? 5;
    const reconnectIntervalMs = streamOptions.reconnectIntervalMs ?? 500;
    const maxReconnectIntervalMs = streamOptions.maxReconnectIntervalMs ?? 10_000;
    const signal = streamOptions.signal;
    const path = `/v1/async_tasks/${encodeURIComponent(taskId)}/stream`;

    let seen = 0; // SSE events already yielded (server replays from 0 on attach)
    let reconnects = 0;

    for (;;) {
      if (signal?.aborted) {
        throw signal.reason ?? new DOMException("Aborted", "AbortError");
      }

      let response: Response;
      try {
        response = await fetchImpl(`${baseURL}${path}`, {
          method: "GET",
          headers: buildHeaders({ Accept: "text/event-stream" }),
          signal: signal ?? null,
        });
      } catch (error) {
        if (isAbortError(error) || signal?.aborted) throw error;
        reconnects += 1;
        if (reconnects > maxReconnects) {
          throw new TensorZeroStreamError(
            `Event stream for task ${taskId} failed after ${maxReconnects} reconnects`,
            { cause: error },
          );
        }
        await sleep(
          Math.min(reconnectIntervalMs * 2 ** (reconnects - 1), maxReconnectIntervalMs),
          signal,
        );
        continue;
      }

      if (response.status === 410) {
        // The event stream is gone (expired TTL or never written). Fall back to
        // polling until the task reaches a terminal state, then end quietly —
        // the caller fetches the final result via getTask/waitForCompletion.
        await waitForCompletion(taskId, {
          intervalMs: streamOptions.fallbackPollIntervalMs ?? 1_000,
          signal,
        });
        return;
      }
      if (!response.ok) {
        const error = await httpErrorFromResponse(response);
        // A 5xx on attach is often transient on the gateway (e.g. the Redis
        // stream read timing out while Valkey is flapping); retry it within
        // the reconnect budget. Permanent server-side config errors — async
        // inference disabled, or Valkey not configured — are thrown as-is.
        const permanent =
          error instanceof AsyncInferenceDisabledError ||
          (error.statusCode === 500 && /requires Valkey/i.test(error.message)) ||
          error.statusCode < 500;
        if (permanent) throw error;
        reconnects += 1;
        if (reconnects > maxReconnects) {
          throw new TensorZeroStreamError(
            `Event stream for task ${taskId} failed after ${maxReconnects} reconnects`,
            { cause: error },
          );
        }
        await sleep(
          Math.min(reconnectIntervalMs * 2 ** (reconnects - 1), maxReconnectIntervalMs),
          signal,
        );
        continue;
      }
      if (!response.body) {
        throw new TensorZeroStreamError(
          `Event stream response for task ${taskId} has no body`,
        );
      }

      try {
        // The server replays the full history on every attach; skip the events
        // already yielded before a reconnect.
        let received = 0;
        for await (const raw of parseSseStream(response.body)) {
          if (received < seen) {
            received += 1;
            continue;
          }
          const sequence = seen;
          seen += 1;
          received += 1;
          yield {
            type: "event",
            sequence,
            event: raw.event,
            data: raw.data,
            json: tryParseJson(raw.data),
          };
        }
      } catch (error) {
        if (isAbortError(error) || signal?.aborted) throw error;
        reconnects += 1;
        if (reconnects > maxReconnects) {
          throw new TensorZeroStreamError(
            `Event stream for task ${taskId} failed after ${maxReconnects} reconnects`,
            { cause: error },
          );
        }
        await sleep(
          Math.min(reconnectIntervalMs * 2 ** (reconnects - 1), maxReconnectIntervalMs),
          signal,
        );
        continue;
      }

      // The SSE stream ended (terminal `done`/`error` marker, or the task
      // reached a terminal state without one). Re-check the task status once
      // to decide between ending the iteration and reconnecting; the status
      // itself is never yielded — the wire stream is over either way.
      const status = await getTask(taskId);
      if (isTerminalTaskStatus(status)) {
        return;
      }
    }
  }

  const submit = async (
    kind: AsyncApiKind,
    body: AsyncRequestBody,
  ): Promise<AsyncInferenceLaunch> => {
    const raw = (await requestJson("POST", SUBMIT_PATHS[kind], { body })) as Record<
      string,
      unknown
    >;
    if (typeof raw["task_id"] !== "string") {
      throw new TensorZeroParseError("Submit response is missing `task_id`", raw);
    }
    return { taskId: raw["task_id"] };
  };

  return {
    submit,
    submitChatCompletion: (body) => submit("chat", body),
    submitResponses: (body) => submit("responses", body),
    submitMessages: (body) => submit("messages", body),
    getTask,
    streamTask: (taskId, streamOptions = {}) => streamEvents(taskId, streamOptions),
    waitForCompletion,
    status: () => requestJson("GET", "/status"),
    health: () => requestJson("GET", "/health"),
  };
}

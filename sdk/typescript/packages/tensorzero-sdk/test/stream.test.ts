// Modified by Delta-AI under Apache 2.0
import { describe, expect, it } from "vitest";
import { createTensorZeroClient } from "../src/index.js";

function sseBody(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    async start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
        // Yield to the event loop so consumers observe backpressure boundaries.
        await Promise.resolve();
      }
      controller.close();
    },
  });
}

function failingStream(afterChunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    async start(controller) {
      for (const chunk of afterChunks) {
        controller.enqueue(encoder.encode(chunk));
        await Promise.resolve();
      }
      controller.error(new Error("connection reset"));
    },
  });
}

describe("streamTask", () => {
  it("yields exactly the wire frames, then ends silently at EOF", async () => {
    let statusCalls = 0;
    const fetchImpl = (async (input: string | URL | Request) => {
      const url = String(input);
      if (url.endsWith("/stream")) {
        return new Response(
          sseBody([
            'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n',
            'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n',
            "data: [DONE]\n\n",
          ]),
          { status: 200, headers: { "Content-Type": "text/event-stream" } },
        );
      }
      statusCalls += 1;
      return new Response(
        JSON.stringify({ status: "completed", task_id: "t1", response: { id: "x" } }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    }) as typeof globalThis.fetch;

    const client = createTensorZeroClient({
      baseURL: "http://gateway.test",
      fetch: fetchImpl,
    });
    const items = [];
    for await (const item of client.streamTask("t1")) {
      items.push(item);
    }

    // Exactly the frames the mock wire sent — no synthetic terminal item.
    expect(items).toHaveLength(3);
    expect(items[0]).toMatchObject({
      type: "event",
      sequence: 0,
      event: undefined,
      json: { choices: [{ delta: { content: "Hel" } }] },
    });
    expect(items[1]).toMatchObject({ type: "event", sequence: 1 });
    expect(items[2]).toMatchObject({ type: "event", sequence: 2, data: "[DONE]" });
    expect(items[2]!.json).toBeUndefined();
    // getTask is called once internally to decide end-vs-reconnect; the
    // result is never yielded.
    expect(statusCalls).toBe(1);
  });

  it("passes through the wire error frame, then ends", async () => {
    const fetchImpl = (async (input: string | URL | Request) => {
      const url = String(input);
      if (url.endsWith("/stream")) {
        return new Response(
          sseBody([
            'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n',
            'event: error\ndata: {"error":{"message":"provider exploded"}}\n\n',
          ]),
          { status: 200 },
        );
      }
      return new Response(
        JSON.stringify({
          status: "failed",
          task_id: "t1",
          error: { message: "provider exploded" },
        }),
        { status: 200 },
      );
    }) as typeof globalThis.fetch;

    const client = createTensorZeroClient({
      baseURL: "http://gateway.test",
      fetch: fetchImpl,
    });
    const items = [];
    for await (const item of client.streamTask("t1")) items.push(item);

    expect(items).toHaveLength(2);
    expect(items[1]).toMatchObject({
      type: "event",
      event: "error",
      json: { error: { message: "provider exploded" } },
    });
  });

  it("reconnects after a broken stream and deduplicates replayed events", async () => {
    let streamCalls = 0;
    const fetchImpl = (async (input: string | URL | Request) => {
      const url = String(input);
      if (url.endsWith("/stream")) {
        streamCalls += 1;
        if (streamCalls === 1) {
          // First attach: two events, then the connection drops.
          return new Response(
            failingStream([
              'data: {"n":0}\n\n',
              'data: {"n":1}\n\n',
            ]),
            { status: 200 },
          );
        }
        // Reconnect: server replays the full history plus the new event.
        return new Response(
          sseBody(['data: {"n":0}\n\n', 'data: {"n":1}\n\n', 'data: {"n":2}\n\n']),
          { status: 200 },
        );
      }
      return new Response(
        JSON.stringify({ status: "completed", task_id: "t1", response: {} }),
        { status: 200 },
      );
    }) as typeof globalThis.fetch;

    const client = createTensorZeroClient({
      baseURL: "http://gateway.test",
      fetch: fetchImpl,
    });
    const items = [];
    for await (const item of client.streamTask("t1", { reconnectIntervalMs: 1 })) {
      items.push(item);
    }

    // Exactly one copy of each wire event, in order; the stream then ends
    // without any synthetic trailing item.
    expect(items.map((i) => i.data)).toEqual(['{"n":0}', '{"n":1}', '{"n":2}']);
    expect(items.map((i) => i.sequence)).toEqual([0, 1, 2]);
    expect(streamCalls).toBe(2);
  });

  it("falls back to polling on 410 and ends without yielding anything", async () => {
    let statusCalls = 0;
    const fetchImpl = (async (input: string | URL | Request) => {
      const url = String(input);
      if (url.endsWith("/stream")) {
        return new Response(
          JSON.stringify({ error: { message: "stream is no longer available" } }),
          { status: 410 },
        );
      }
      statusCalls += 1;
      const body =
        statusCalls === 1
          ? { status: "running", task_id: "t1", elapsed_ms: 10 }
          : { status: "completed", task_id: "t1", response: { id: "late" } };
      return new Response(JSON.stringify(body), { status: 200 });
    }) as typeof globalThis.fetch;

    const client = createTensorZeroClient({
      baseURL: "http://gateway.test",
      fetch: fetchImpl,
    });
    const items = [];
    for await (const item of client.streamTask("t1", {
      fallbackPollIntervalMs: 1,
    })) {
      items.push(item);
    }

    // Polling reached a terminal state, and nothing was yielded: the 410
    // fallback produces no wire frames, so the iterable is simply empty.
    expect(items).toEqual([]);
    expect(statusCalls).toBe(2);
  });

  it("re-attaches when the stream hits EOF but the task is still running", async () => {
    let streamCalls = 0;
    let statusCalls = 0;
    const fetchImpl = (async (input: string | URL | Request) => {
      const url = String(input);
      if (url.endsWith("/stream")) {
        streamCalls += 1;
        if (streamCalls === 1) {
          // First attach: one event, then a bare EOF (no terminal marker).
          return new Response(sseBody(['data: {"n":0}\n\n']), { status: 200 });
        }
        // Re-attach: full replay plus the remaining events.
        return new Response(
          sseBody(['data: {"n":0}\n\n', 'data: {"n":1}\n\n', "data: [DONE]\n\n"]),
          { status: 200 },
        );
      }
      statusCalls += 1;
      const body =
        statusCalls === 1
          ? { status: "running", task_id: "t1", elapsed_ms: 5 }
          : { status: "completed", task_id: "t1", response: {} };
      return new Response(JSON.stringify(body), { status: 200 });
    }) as typeof globalThis.fetch;

    const client = createTensorZeroClient({
      baseURL: "http://gateway.test",
      fetch: fetchImpl,
    });
    const items = [];
    for await (const item of client.streamTask("t1", { reconnectIntervalMs: 1 })) {
      items.push(item);
    }

    expect(items.map((i) => i.data)).toEqual(['{"n":0}', '{"n":1}', "[DONE]"]);
    expect(streamCalls).toBe(2);
    expect(statusCalls).toBe(2);
  });

  it("retries a transient 5xx on attach, then streams normally", async () => {
    let streamCalls = 0;
    const fetchImpl = (async (input: string | URL | Request) => {
      const url = String(input);
      if (url.endsWith("/stream")) {
        streamCalls += 1;
        if (streamCalls === 1) {
          // Gateway-side transient failure (e.g. Valkey read timeout).
          return new Response(
            JSON.stringify({
              error: { message: "Internal error: Failed to read async inference event stream: timed out" },
            }),
            { status: 500 },
          );
        }
        return new Response(sseBody(['data: {"n":0}\n\n']), { status: 200 });
      }
      return new Response(
        JSON.stringify({ status: "completed", task_id: "t1", response: {} }),
        { status: 200 },
      );
    }) as typeof globalThis.fetch;

    const client = createTensorZeroClient({
      baseURL: "http://gateway.test",
      fetch: fetchImpl,
    });
    const items = [];
    for await (const item of client.streamTask("t1", { reconnectIntervalMs: 1 })) {
      items.push(item);
    }
    expect(items.map((i) => i.data)).toEqual(['{"n":0}']);
    expect(streamCalls).toBe(2);
  });

  it("does not retry permanent 500 config errors", async () => {
    const fetchImpl = (async () =>
      new Response(
        JSON.stringify({
          error: { message: "Async inference is not enabled. Set gateway.async_inference.enabled = true." },
        }),
        { status: 500 },
      )) as typeof globalThis.fetch;
    const client = createTensorZeroClient({
      baseURL: "http://gateway.test",
      fetch: fetchImpl,
    });
    await expect(async () => {
      for await (const _ of client.streamTask("t1", { reconnectIntervalMs: 1 })) {
        // consume
      }
    }).rejects.toMatchObject({ name: "AsyncInferenceDisabledError", statusCode: 500 });
  });

  it("throws TensorZeroStreamError when reconnects are exhausted", async () => {
    const fetchImpl = (async (input: string | URL | Request) => {
      if (String(input).endsWith("/stream")) {
        throw new Error("connection refused");
      }
      return new Response("{}", { status: 200 });
    }) as typeof globalThis.fetch;

    const client = createTensorZeroClient({
      baseURL: "http://gateway.test",
      fetch: fetchImpl,
    });
    await expect(async () => {
      for await (const _ of client.streamTask("t1", {
        maxReconnects: 2,
        reconnectIntervalMs: 1,
      })) {
        // consume
      }
    }).rejects.toMatchObject({ name: "TensorZeroStreamError" });
  });

  it("throws TaskNotFoundError on 404 without retrying", async () => {
    const fetchImpl = (async () =>
      new Response(JSON.stringify({ error: { message: "Unknown route" } }), {
        status: 404,
      })) as typeof globalThis.fetch;
    const client = createTensorZeroClient({
      baseURL: "http://gateway.test",
      fetch: fetchImpl,
    });
    await expect(async () => {
      for await (const _ of client.streamTask("nope")) {
        // consume
      }
    }).rejects.toMatchObject({ name: "TaskNotFoundError" });
  });

  it("respects the abort signal mid-stream", async () => {
    const controller = new AbortController();
    const fetchImpl = (async (input: string | URL | Request, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/stream")) {
        return new Response(
          new ReadableStream({
            start(streamController) {
              const encoder = new TextEncoder();
              streamController.enqueue(encoder.encode('data: {"n":0}\n\n'));
              init?.signal?.addEventListener("abort", () => {
                streamController.error(
                  new DOMException("Aborted", "AbortError"),
                );
              });
            },
          }),
          { status: 200 },
        );
      }
      return new Response("{}", { status: 200 });
    }) as typeof globalThis.fetch;

    const client = createTensorZeroClient({
      baseURL: "http://gateway.test",
      fetch: fetchImpl,
    });
    setTimeout(() => controller.abort(), 20);
    await expect(async () => {
      for await (const _ of client.streamTask("t1", {
        signal: controller.signal,
        reconnectIntervalMs: 1,
      })) {
        // consume
      }
    }).rejects.toMatchObject({ name: "AbortError" });
  });
});

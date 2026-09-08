// Modified by Delta-AI under Apache 2.0
import { describe, expect, it } from "vitest";
import { createTensorZeroClient } from "../src/index.js";

interface RecordedRequest {
  url: string;
  method: string;
  headers: Record<string, string>;
  body: unknown;
}

function mockFetch(
  handler: (request: RecordedRequest) => { status?: number; body?: unknown },
) {
  const requests: RecordedRequest[] = [];
  const fetchImpl = (async (
    input: string | URL | Request,
    init?: RequestInit,
  ): Promise<Response> => {
    const headers: Record<string, string> = {};
    for (const [key, value] of Object.entries(init?.headers ?? {})) {
      headers[key] = value as string;
    }
    const request: RecordedRequest = {
      url: String(input),
      method: init?.method ?? "GET",
      headers,
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
    };
    requests.push(request);
    const { status = 200, body } = handler(request);
    return new Response(body === undefined ? null : JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof globalThis.fetch;
  return { fetchImpl, requests };
}

const baseOptions = { baseURL: "http://gateway.test/", apiKey: "sk-test" };

describe("submit", () => {
  it("posts to the kind-specific path and returns taskId", async () => {
    const { fetchImpl, requests } = mockFetch(() => ({
      status: 202,
      body: { task_id: "0190f9c4-8e3a-7b3d-9c1e-2f4a5b6c7d8e" },
    }));
    const client = createTensorZeroClient({ ...baseOptions, fetch: fetchImpl });

    const launch = await client.submitChatCompletion({
      model: "openai::gpt-5",
      messages: [{ role: "user", content: "hi" }],
    });

    expect(launch.taskId).toBe("0190f9c4-8e3a-7b3d-9c1e-2f4a5b6c7d8e");
    expect(requests[0]!.url).toBe("http://gateway.test/v1/chat/completions/async");
    expect(requests[0]!.method).toBe("POST");
    expect(requests[0]!.headers["Authorization"]).toBe("Bearer sk-test");
    expect(requests[0]!.body).toEqual({
      model: "openai::gpt-5",
      messages: [{ role: "user", content: "hi" }],
    });
  });

  it("maps kind to the right path", async () => {
    const { fetchImpl, requests } = mockFetch(() => ({
      status: 202,
      body: { task_id: "t" },
    }));
    const client = createTensorZeroClient({ ...baseOptions, fetch: fetchImpl });
    await client.submit("chat", {});
    await client.submit("responses", {});
    await client.submit("messages", {});
    expect(requests.map((r) => r.url)).toEqual([
      "http://gateway.test/v1/chat/completions/async",
      "http://gateway.test/v1/responses/async",
      "http://gateway.test/v1/messages/async",
    ]);
  });
});

describe("getTask", () => {
  it.each([
    [
      { status: "queued", task_id: "t1", queue_position: 3 },
      { status: "queued", taskId: "t1", queuePosition: 3 },
    ],
    [
      { status: "queued", task_id: "t1" },
      { status: "queued", taskId: "t1" },
    ],
    [
      {
        status: "running",
        task_id: "t1",
        started_at: "2026-09-03T10:00:00Z",
        elapsed_ms: 1500,
      },
      {
        status: "running",
        taskId: "t1",
        startedAt: "2026-09-03T10:00:00Z",
        elapsedMs: 1500,
      },
    ],
    [
      { status: "completed", task_id: "t1", response: { id: "chatcmpl-123" } },
      { status: "completed", taskId: "t1", response: { id: "chatcmpl-123" } },
    ],
    [{ status: "failed", task_id: "t1" }, { status: "failed", taskId: "t1" }],
    [
      { status: "cancelled", task_id: "t1", error: { message: "boom" } },
      { status: "cancelled", taskId: "t1", error: { message: "boom" } },
    ],
  ] as const)("parses %j", async (wire, expected) => {
    const { fetchImpl } = mockFetch(() => ({ body: wire }));
    const client = createTensorZeroClient({ ...baseOptions, fetch: fetchImpl });
    expect(await client.getTask("t1")).toEqual(expected);
  });

  it("throws TaskNotFoundError on 404", async () => {
    const { fetchImpl } = mockFetch(() => ({
      status: 404,
      body: { error: { message: "Unknown route" } },
    }));
    const client = createTensorZeroClient({ ...baseOptions, fetch: fetchImpl });
    await expect(client.getTask("nope")).rejects.toMatchObject({
      name: "TaskNotFoundError",
      statusCode: 404,
    });
  });

  it("throws AsyncInferenceDisabledError on 500 'not enabled'", async () => {
    const { fetchImpl } = mockFetch(() => ({
      status: 500,
      body: { error: { message: "Async inference is not enabled." } },
    }));
    const client = createTensorZeroClient({ ...baseOptions, fetch: fetchImpl });
    await expect(client.getTask("t1")).rejects.toMatchObject({
      name: "AsyncInferenceDisabledError",
      statusCode: 500,
    });
  });

  it("throws TensorZeroHttpError for other status codes", async () => {
    const { fetchImpl } = mockFetch(() => ({
      status: 502,
      body: { error: { message: "bad gateway" } },
    }));
    const client = createTensorZeroClient({ ...baseOptions, fetch: fetchImpl });
    await expect(client.getTask("t1")).rejects.toMatchObject({
      name: "TensorZeroHttpError",
      statusCode: 502,
    });
  });

  it("throws TensorZeroParseError on unknown status tag", async () => {
    const { fetchImpl } = mockFetch(() => ({
      body: { status: "mysterious", task_id: "t1" },
    }));
    const client = createTensorZeroClient({ ...baseOptions, fetch: fetchImpl });
    await expect(client.getTask("t1")).rejects.toMatchObject({
      name: "TensorZeroParseError",
    });
  });
});

describe("waitForCompletion", () => {
  it("polls until a terminal state with exponential backoff", async () => {
    const statuses = [
      { status: "queued", task_id: "t1", queue_position: 2 },
      { status: "running", task_id: "t1", elapsed_ms: 42 },
      { status: "completed", task_id: "t1", response: { id: "done" } },
    ];
    let calls = 0;
    const { fetchImpl } = mockFetch(() => ({ body: statuses[calls++] }));
    const client = createTensorZeroClient({ ...baseOptions, fetch: fetchImpl });

    const start = Date.now();
    const result = await client.waitForCompletion("t1", {
      intervalMs: 10,
      maxIntervalMs: 25,
    });
    const elapsed = Date.now() - start;

    expect(result).toEqual({
      status: "completed",
      taskId: "t1",
      response: { id: "done" },
    });
    expect(calls).toBe(3);
    // backoff: 10ms then min(20ms, 25ms) = 20ms
    expect(elapsed).toBeGreaterThanOrEqual(25);
  });

  it("treats failed and cancelled as terminal", async () => {
    const { fetchImpl } = mockFetch(() => ({
      body: { status: "failed", task_id: "t1", error: { message: "boom" } },
    }));
    const client = createTensorZeroClient({ ...baseOptions, fetch: fetchImpl });
    const result = await client.waitForCompletion("t1", { intervalMs: 1 });
    expect(result.status).toBe("failed");
  });

  it("times out with TensorZeroTimeoutError", async () => {
    const { fetchImpl } = mockFetch(() => ({
      body: { status: "queued", task_id: "t1" },
    }));
    const client = createTensorZeroClient({ ...baseOptions, fetch: fetchImpl });
    await expect(
      client.waitForCompletion("t1", { intervalMs: 5, timeoutMs: 20 }),
    ).rejects.toMatchObject({ name: "TensorZeroTimeoutError" });
  });

  it("respects the abort signal", async () => {
    const { fetchImpl } = mockFetch(() => ({
      body: { status: "queued", task_id: "t1" },
    }));
    const client = createTensorZeroClient({ ...baseOptions, fetch: fetchImpl });
    const controller = new AbortController();
    setTimeout(() => controller.abort(), 10);
    await expect(
      client.waitForCompletion("t1", {
        intervalMs: 1_000,
        signal: controller.signal,
      }),
    ).rejects.toMatchObject({ name: "AbortError" });
  });
});

describe("status/health", () => {
  it("hits /status and /health", async () => {
    const { fetchImpl, requests } = mockFetch(() => ({ body: { status: "ok" } }));
    const client = createTensorZeroClient({ ...baseOptions, fetch: fetchImpl });
    expect(await client.status()).toEqual({ status: "ok" });
    expect(await client.health()).toEqual({ status: "ok" });
    expect(requests.map((r) => r.url)).toEqual([
      "http://gateway.test/status",
      "http://gateway.test/health",
    ]);
  });
});

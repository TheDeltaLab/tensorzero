// Modified by Delta-AI under Apache 2.0
import { describe, expect, it } from "vitest";
import type { LanguageModelV4Prompt } from "@ai-sdk/provider";
import { createTensorZero } from "../src/index.js";
import { decodeBatchId } from "../src/batch-reference.js";

interface RecordedRequest {
  url: string;
  method: string;
  body: unknown;
}

function gatewayFetch(
  handler: (request: RecordedRequest) => { status?: number; body?: unknown },
) {
  const requests: RecordedRequest[] = [];
  const fetchImpl = (async (input: string | URL | Request, init?: RequestInit) => {
    const request: RecordedRequest = {
      url: String(input),
      method: init?.method ?? "GET",
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

const prompt = (text: string): LanguageModelV4Prompt => [
  { role: "user", content: [{ type: "text", text }] },
];

const baseSettings = {
  baseURL: "http://gateway.test/v1",
  apiKey: "sk-test",
};

describe("createTensorZero", () => {
  it("returns a v4 provider whose chat model implements the batch capability", () => {
    const provider = createTensorZero({
      ...baseSettings,
      fetch: gatewayFetch(() => ({ body: {} })).fetchImpl,
    });
    const model = provider("openai::gpt-5");
    expect(model.specificationVersion).toBe("v4");
    expect(model.provider).toBe("tensorzero.chat");
    expect(model.modelId).toBe("openai::gpt-5");
    expect(typeof model.doGenerate).toBe("function");
    expect(typeof model.doStream).toBe("function");
    expect(typeof model.experimental_doStartBatch).toBe("function");
    expect(typeof model.experimental_doGetBatchStatus).toBe("function");
    expect(typeof model.experimental_doGetBatchResults).toBe("function");
    expect(provider.asyncClient).toBeDefined();
  });
});

describe("experimental_doStartBatch", () => {
  it("serializes each request via the chat model and submits async tasks in parallel", async () => {
    let submitCount = 0;
    const { fetchImpl, requests } = gatewayFetch(() => ({
      status: 202,
      body: { task_id: `task-${submitCount++}` },
    }));
    const provider = createTensorZero({ ...baseSettings, fetch: fetchImpl });
    const model = provider("openai::gpt-5");

    const result = await model.experimental_doStartBatch({
      requests: [
        { id: "req-a", options: { prompt: prompt("hello") } },
        {
          id: "req-b",
          options: {
            prompt: prompt("world"),
            temperature: 0.5,
            providerOptions: { tensorzero: { cache_options: { enabled: true } } },
          },
        },
      ],
    });

    expect(result.status).toBe("pending");
    expect(result.requestCounts).toEqual({
      total: 2,
      pending: 2,
      completed: 0,
      failed: 0,
    });
    expect(result.warnings).toEqual([]);

    const reference = decodeBatchId(result.batchId);
    expect(reference).toEqual({
      v: 1,
      items: [
        { id: "req-a", taskId: "task-0" },
        { id: "req-b", taskId: "task-1" },
      ],
    });

    // Both submissions hit the chat completions async endpoint with the body
    // the sync chat path would have sent.
    const submits = requests.filter((r) => r.url.endsWith("/chat/completions/async"));
    expect(submits).toHaveLength(2);
    expect(submits[0]!.method).toBe("POST");
    const bodyA = submits[0]!.body as Record<string, unknown>;
    expect(bodyA["model"]).toBe("openai::gpt-5");
    expect(bodyA["messages"]).toEqual([{ role: "user", content: "hello" }]);
    const bodyB = submits[1]!.body as Record<string, unknown>;
    expect(bodyB["temperature"]).toBe(0.5);
    // Gateway-private options pass through into the request body.
    expect(bodyB["cache_options"]).toEqual({ enabled: true });
  });

  it("warns that completion webhooks are unsupported", async () => {
    const { fetchImpl } = gatewayFetch(() => ({ status: 202, body: { task_id: "t" } }));
    const provider = createTensorZero({ ...baseSettings, fetch: fetchImpl });
    const result = await provider("m").experimental_doStartBatch({
      requests: [{ id: "r1", options: { prompt: prompt("hi") } }],
      webhookUrl: "https://example.com/hook",
    });
    expect(result.warnings[0]!.warning).toMatchObject({
      type: "unsupported",
      feature: "webhookUrl",
    });
  });
});

describe("experimental_doGetBatchStatus", () => {
  function statusFetch(statusByTaskId: Record<string, unknown>) {
    return gatewayFetch((request) => {
      const match = /\/v1\/async_tasks\/([^/]+)$/.exec(request.url);
      const taskId = match?.[1] ?? "";
      return { body: statusByTaskId[taskId] };
    });
  }

  const batchId = JSON.stringify({
    v: 1,
    items: [
      { id: "req-a", taskId: "task-0" },
      { id: "req-b", taskId: "task-1" },
      { id: "req-c", taskId: "task-2" },
    ],
  });

  it("maps queued/running tasks to pending", async () => {
    const { fetchImpl } = statusFetch({
      "task-0": { status: "completed", task_id: "task-0", response: {} },
      "task-1": { status: "running", task_id: "task-1", elapsed_ms: 5 },
      "task-2": { status: "queued", task_id: "task-2", queue_position: 1 },
    });
    const provider = createTensorZero({ ...baseSettings, fetch: fetchImpl });
    const status = await provider("m").experimental_doGetBatchStatus({ batchId });
    expect(status.status).toBe("pending");
    expect(status.requestCounts).toEqual({
      total: 3,
      pending: 2,
      completed: 1,
      failed: 0,
    });
    expect(status.error).toBeUndefined();
  });

  it("maps all-completed to completed", async () => {
    const { fetchImpl } = statusFetch({
      "task-0": { status: "completed", task_id: "task-0", response: {} },
      "task-1": { status: "completed", task_id: "task-1", response: {} },
      "task-2": { status: "completed", task_id: "task-2", response: {} },
    });
    const provider = createTensorZero({ ...baseSettings, fetch: fetchImpl });
    const status = await provider("m").experimental_doGetBatchStatus({ batchId });
    expect(status.status).toBe("completed");
  });

  it("maps any failed/cancelled terminal task to failed with the first error", async () => {
    const { fetchImpl } = statusFetch({
      "task-0": { status: "completed", task_id: "task-0", response: {} },
      "task-1": { status: "failed", task_id: "task-1", error: { message: "boom" } },
      "task-2": { status: "cancelled", task_id: "task-2" },
    });
    const provider = createTensorZero({ ...baseSettings, fetch: fetchImpl });
    const status = await provider("m").experimental_doGetBatchStatus({ batchId });
    expect(status.status).toBe("failed");
    expect(status.requestCounts).toEqual({
      total: 3,
      pending: 0,
      completed: 1,
      failed: 2,
    });
    expect(status.error).toEqual({ message: "boom" });
  });
});

describe("experimental_doGetBatchResults", () => {
  it("streams one item result per task with converted generate results", async () => {
    const chatCompletion = {
      id: "chatcmpl-1",
      created: 1757000000,
      model: "tensorzero::openai::gpt-5",
      choices: [
        {
          finish_reason: "stop",
          message: { role: "assistant", content: "Hello back" },
        },
      ],
      usage: {
        prompt_tokens: 12,
        completion_tokens: 3,
        prompt_tokens_details: { cached_tokens: 4 },
        completion_tokens_details: { reasoning_tokens: 0 },
      },
    };
    const { fetchImpl } = gatewayFetch((request) => {
      const match = /\/v1\/async_tasks\/([^/]+)$/.exec(request.url);
      const taskId = match?.[1] ?? "";
      if (taskId === "task-0") {
        return {
          body: { status: "completed", task_id: taskId, response: chatCompletion },
        };
      }
      return {
        body: { status: "failed", task_id: taskId, error: { message: "nope" } },
      };
    });
    const provider = createTensorZero({ ...baseSettings, fetch: fetchImpl });

    const batchId = JSON.stringify({
      v: 1,
      items: [
        { id: "req-a", taskId: "task-0" },
        { id: "req-b", taskId: "task-1" },
      ],
    });
    const stream = await provider("m").experimental_doGetBatchResults({ batchId });
    const items = [];
    const reader = stream.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      items.push(value);
    }

    expect(items).toHaveLength(2);

    const succeeded = items[0]!;
    expect(succeeded.id).toBe("req-a");
    expect(succeeded.status).toBe("succeeded");
    if (succeeded.status === "succeeded") {
      expect(succeeded.result.content).toEqual([
        { type: "text", text: "Hello back" },
      ]);
      expect(succeeded.result.finishReason).toEqual({
        unified: "stop",
        raw: "stop",
      });
      expect(succeeded.result.usage.inputTokens).toEqual({
        total: 12,
        noCache: 8,
        cacheRead: 4,
        cacheWrite: undefined,
      });
      expect(succeeded.result.response?.modelId).toBe(
        "tensorzero::openai::gpt-5",
      );
    }

    const failed = items[1]!;
    expect(failed).toEqual({
      id: "req-b",
      status: "failed",
      error: { message: "nope" },
    });
  });
});

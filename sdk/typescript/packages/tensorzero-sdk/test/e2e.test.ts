// Modified by Delta-AI under Apache 2.0
import { describe, expect, it } from "vitest";
import { createTensorZeroClient, TaskNotFoundError } from "../src/index.js";

const GATEWAY = process.env["TZ_E2E_GATEWAY"];
const API_KEY = process.env["TZ_E2E_KEY"];
const RUN_E2E = Boolean(GATEWAY && API_KEY);
const MODEL = "deepseek-v4-flash";

const client = RUN_E2E
  ? createTensorZeroClient({ baseURL: GATEWAY!, apiKey: API_KEY! })
  : undefined;

describe.skipIf(!RUN_E2E)("e2e against a real gateway", () => {
  it("status() and health() are reachable without auth failures", async () => {
    // /status is the liveness endpoint and returns 200 when the gateway is up.
    const status = (await client!.status()) as { status?: string };
    expect(status.status).toBe("ok");
    // /health reports per-component health and returns 503 when any component
    // (e.g. valkey) is degraded — both outcomes are valid wire behavior.
    try {
      await client!.health();
    } catch (error) {
      expect(error).toMatchObject({ name: "TensorZeroHttpError", statusCode: 503 });
      expect((error as { body?: { gateway?: string } }).body?.gateway).toBe("ok");
    }
  });

  it("submit → waitForCompletion → completed chat completion", async () => {
    const { taskId } = await client!.submitChatCompletion({
      model: MODEL,
      messages: [{ role: "user", content: "Say hello in one word." }],
      max_tokens: 32,
    });
    expect(typeof taskId).toBe("string");

    const final = await client!.waitForCompletion(taskId, {
      intervalMs: 500,
      maxIntervalMs: 2_000,
      timeoutMs: 120_000,
    });
    expect(final.status).toBe("completed");
    if (final.status !== "completed") return;

    const response = final.response as {
      object?: string;
      choices?: Array<{ message?: { content?: string } }>;
    };
    expect(response.object).toBe("chat.completion");
    const content = response.choices?.[0]?.message?.content;
    expect(typeof content).toBe("string");
    expect(content!.length).toBeGreaterThan(0);
  }, 150_000);

  it("getTask reports legal intermediate states", async () => {
    const { taskId } = await client!.submitChatCompletion({
      model: MODEL,
      messages: [{ role: "user", content: "Count to ten slowly." }],
      max_tokens: 64,
    });

    // Poll directly and collect every state observed along the way.
    const seen: string[] = [];
    for (;;) {
      const status = await client!.getTask(taskId);
      expect(status.taskId).toBe(taskId);
      seen.push(status.status);
      if (status.status === "completed") break;
      if (status.status === "failed" || status.status === "cancelled") {
        throw new Error(`task unexpectedly ${status.status}`);
      }
      await new Promise((resolve) => setTimeout(resolve, 200));
    }

    expect(seen.at(-1)).toBe("completed");
    for (const state of seen) {
      expect(["queued", "running", "completed"]).toContain(state);
    }
  }, 150_000);

  it("streamTask relays wire chunks and ends quietly", async () => {
    const { taskId } = await client!.submitChatCompletion({
      model: MODEL,
      messages: [{ role: "user", content: "Say hello in one word." }],
      max_tokens: 32,
    });

    const events = [];
    for await (const event of client!.streamTask(taskId, {
      reconnectIntervalMs: 200,
      fallbackPollIntervalMs: 200,
    })) {
      events.push(event);
    }

    // Wire-faithful: every yielded item is an "event" frame from the wire;
    // no synthetic terminal item exists in the type or at runtime.
    expect(events.length).toBeGreaterThan(0);
    let sawChunk = false;
    let sawDone = false;
    for (const event of events) {
      expect(event.type).toBe("event");
      if (event.data === "[DONE]") {
        sawDone = true;
        continue;
      }
      const parsed = event.json as { object?: string } | undefined;
      expect(parsed).toBeDefined();
      expect(parsed!.object).toBe("chat.completion.chunk");
      sawChunk = true;
    }
    expect(sawChunk).toBe(true);
    expect(sawDone).toBe(true);

    // The final result comes from getTask, not the stream.
    const status = await client!.getTask(taskId);
    expect(status.status).toBe("completed");
  }, 150_000);

  it("getTask on an unknown task id throws TaskNotFoundError", async () => {
    await expect(
      client!.getTask("0190f9c4-8e3a-7b3d-9c1e-2f4a5b6c7d8e"),
    ).rejects.toBeInstanceOf(TaskNotFoundError);
  });

  it("a bad model name reaches a failed terminal state", async () => {
    const { taskId } = await client!.submitChatCompletion({
      model: "definitely-not-a-model",
      messages: [{ role: "user", content: "hi" }],
      max_tokens: 8,
    });

    const final = await client!.waitForCompletion(taskId, {
      intervalMs: 500,
      maxIntervalMs: 2_000,
      timeoutMs: 60_000,
    });
    expect(final.status).toBe("failed");
    if (final.status !== "failed") return;
    const message = JSON.stringify(final.error);
    expect(message).toContain("not found in model table");
  }, 90_000);
});

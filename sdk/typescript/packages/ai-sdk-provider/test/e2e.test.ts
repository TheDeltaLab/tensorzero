// Modified by Delta-AI under Apache 2.0
import { describe, expect, it } from "vitest";
import {
  generateText,
  experimental_startTextBatch,
  experimental_getBatchStatus,
  experimental_getBatchResults,
} from "ai";
import { createTensorZero } from "../src/index.js";

const GATEWAY = process.env["TZ_E2E_GATEWAY"];
const API_KEY = process.env["TZ_E2E_KEY"];
const RUN_E2E = Boolean(GATEWAY && API_KEY);
const MODEL = "deepseek-v4-flash";

const tensorzero = RUN_E2E
  ? createTensorZero({ baseURL: `${GATEWAY}/v1`, apiKey: API_KEY! })
  : undefined;

describe.skipIf(!RUN_E2E)("e2e against a real gateway", () => {
  it("generateText works against the gateway's OpenAI-compatible API", async () => {
    const result = await generateText({
      model: tensorzero!(MODEL),
      prompt: "Say hello in one word.",
      maxOutputTokens: 32,
    });
    expect(typeof result.text).toBe("string");
    expect(result.text.length).toBeGreaterThan(0);
  }, 120_000);

  it("startTextBatch → getBatchStatus → getBatchResults end to end", async () => {
    const model = tensorzero!(MODEL);

    const batch = await experimental_startTextBatch({
      model,
      requests: [
        {
          id: "e2e-req-1",
          prompt: "Say hello in one word.",
          maxOutputTokens: 32,
        },
        {
          id: "e2e-req-2",
          prompt: "Say goodbye in one word.",
          maxOutputTokens: 32,
        },
      ],
    });
    expect(batch.id).toBeTruthy();
    expect(batch.status).toBe("pending");

    // The batch reference is serializable; simulate persisting and resuming.
    const restored = JSON.parse(JSON.stringify(batch));

    let status = await experimental_getBatchStatus({ model, batch: restored });
    const deadline = Date.now() + 120_000;
    while (status.status === "pending") {
      if (Date.now() > deadline) {
        throw new Error("batch did not reach a terminal state in time");
      }
      await new Promise((resolve) => setTimeout(resolve, 1_000));
      status = await experimental_getBatchStatus({ model, batch: restored });
    }
    expect(status.status).toBe("completed");
    expect(status.requestCounts).toMatchObject({
      total: 2,
      completed: 2,
      failed: 0,
    });

    const byId = new Map<string, string>();
    for await (const item of experimental_getBatchResults({ model, batch: restored })) {
      expect(item.status).toBe("succeeded");
      if (item.status === "succeeded") {
        expect(item.text.length).toBeGreaterThan(0);
        byId.set(item.id, item.text);
      }
    }
    expect([...byId.keys()].sort()).toEqual(["e2e-req-1", "e2e-req-2"]);
  }, 180_000);
});

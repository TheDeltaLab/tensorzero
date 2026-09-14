// Modified by Delta-AI under Apache 2.0
import { describe, expect, test, beforeAll } from "vitest";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import type { TensorZeroClient } from "~/utils/tensorzero/tensorzero";

let tensorZeroClient: TensorZeroClient;

describe("getInferenceCount", () => {
  beforeAll(() => {
    tensorZeroClient = getTensorZeroClient();
  });

  test("should return inference count for a function", async () => {
    const stats = await tensorZeroClient.getInferenceCount("extract_entities");
    expect(stats.inference_count).toBeGreaterThanOrEqual(604);
  });

  test("should return inference count for a function and variant", async () => {
    const stats = await tensorZeroClient.getInferenceCount("extract_entities", {
      variantName: "gpt4o_initial_prompt",
    });
    expect(stats.inference_count).toBeGreaterThanOrEqual(132);
  });

  test("should throw error for unknown function", async () => {
    await expect(
      tensorZeroClient.getInferenceCount("nonexistent_function"),
    ).rejects.toThrow();
  });

  test("should throw error for unknown variant", async () => {
    await expect(
      tensorZeroClient.getInferenceCount("extract_entities", {
        variantName: "nonexistent_variant",
      }),
    ).rejects.toThrow();
  });
});

describe("getFeedbackCount", () => {
  beforeAll(() => {
    tensorZeroClient = getTensorZeroClient();
  });

  test("should return feedback stats for boolean metric", async () => {
    const stats = await tensorZeroClient.getFeedbackCount(
      "extract_entities",
      "exact_match",
    );
    expect(stats.feedback_count).toBeGreaterThanOrEqual(99);
    expect(stats.inference_count).toBeGreaterThanOrEqual(41);
  });

  test("should return feedback stats for float metric with threshold", async () => {
    const stats = await tensorZeroClient.getFeedbackCount(
      "extract_entities",
      "jaccard_similarity",
      0.8,
    );
    expect(stats.feedback_count).toBeGreaterThanOrEqual(99);
    expect(stats.inference_count).toBeGreaterThanOrEqual(54);
  });

  test("should return feedback stats for demonstration metric", async () => {
    const stats = await tensorZeroClient.getFeedbackCount(
      "extract_entities",
      "demonstration",
    );
    expect(stats.feedback_count).toBeGreaterThanOrEqual(100);
    // For demonstrations, feedback_count equals inference_count
    expect(stats.inference_count).toBe(stats.feedback_count);
  });

  test("should throw error for unknown function", async () => {
    await expect(
      tensorZeroClient.getFeedbackCount("nonexistent_function", "exact_match"),
    ).rejects.toThrow();
  });

  test("should throw error for unknown metric", async () => {
    await expect(
      tensorZeroClient.getFeedbackCount(
        "extract_entities",
        "nonexistent_metric",
      ),
    ).rejects.toThrow();
  });
});

describe("getUsedVariants", () => {
  test("getUsedVariants for extract_entities", async () => {
    const functionName = "extract_entities";
    const result = await tensorZeroClient.getUsedVariants(functionName);
    expect(result).toEqual(
      expect.arrayContaining([
        "baseline",
        "dicl",
        "llama_8b_initial_prompt",
        "gpt4o_mini_initial_prompt",
        "gpt4o_initial_prompt",
        "turbo",
      ]),
    );
    expect(result.length).toBe(6);
  });
});

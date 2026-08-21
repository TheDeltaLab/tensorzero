// Modified by Delta-AI under Apache 2.0
import { describe, expect, test } from "vitest";
import {
  analysisModelsForKind,
  analysisSearchParams,
  formatBucketLabel,
  formatCompactCount,
  formatInputCacheHitDescription,
  parseAnalysisQuery,
  rangeDescription,
} from "./analysisQuery";

describe("parseAnalysisQuery", () => {
  test("defaults to 24h chat", () => {
    expect(parseAnalysisQuery(new URLSearchParams())).toEqual({
      range: "24h",
      kind: "chat",
      apiKey: "",
      model: "",
      cacheMissOnly: false,
    });
  });

  test("reads filters and rejects unknown range/kind", () => {
    const params = new URLSearchParams(
      "range=7d&kind=embedding&api_key=abc&model=deepseek-v4-flash&cache_miss_only=true",
    );
    expect(parseAnalysisQuery(params)).toEqual({
      range: "7d",
      kind: "embedding",
      apiKey: "abc",
      model: "deepseek-v4-flash",
      cacheMissOnly: true,
    });
    expect(
      parseAnalysisQuery(new URLSearchParams("range=year&kind=rerank")),
    ).toEqual({
      range: "24h",
      kind: "chat",
      apiKey: "",
      model: "",
      cacheMissOnly: false,
    });
  });
});

describe("analysisSearchParams", () => {
  test("omits defaults", () => {
    expect(
      analysisSearchParams({
        range: "24h",
        kind: "chat",
        apiKey: "",
        model: "",
        cacheMissOnly: false,
      }).toString(),
    ).toBe("");
    expect(
      analysisSearchParams({
        range: "15m",
        kind: "embedding",
        apiKey: "o6bTIwfcUBKV",
        model: "text-embedding-3-small",
        cacheMissOnly: true,
      }).toString(),
    ).toBe(
      "range=15m&kind=embedding&api_key=o6bTIwfcUBKV&model=text-embedding-3-small&cache_miss_only=true",
    );
  });
});

describe("formatters", () => {
  test("compact counts and range copy", () => {
    expect(formatCompactCount(12)).toBe("12");
    expect(formatCompactCount(1500)).toBe("1.5K");
    expect(formatCompactCount(2_300_000)).toBe("2.3M");
    expect(rangeDescription("24h")).toBe("Last 24 hours");
    expect(formatInputCacheHitDescription(2500, 10_000)).toBe(
      "2.5K / 10.0K input tokens",
    );
  });

  test("bucket labels follow Synapse minute/hour/day rules", () => {
    expect(formatBucketLabel("2026-08-21")).toBe("Aug 21");
    expect(formatBucketLabel("2026-08-21T05:07:00Z")).toMatch(/\d/);
    expect(formatBucketLabel("not-a-date")).toBe("not-a-date");
  });
});

describe("analysisModelsForKind", () => {
  const config = {
    model_names: ["gpt-4o"],
    embedding_model_names: ["text-embedding-3-small"],
    model_aliases: [{ name: "deepseek-v4-flash" }],
  };

  test("chat includes aliases and model names", () => {
    expect(analysisModelsForKind("chat", config)).toEqual([
      "deepseek-v4-flash",
      "gpt-4o",
    ]);
  });

  test("embedding uses embedding models only", () => {
    expect(analysisModelsForKind("embedding", config)).toEqual([
      "text-embedding-3-small",
    ]);
  });
});

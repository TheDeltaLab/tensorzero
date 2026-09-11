// Modified by Delta-AI under Apache 2.0
import { describe, expect, test } from "vitest";
import {
  analysisModelsForKind,
  analysisSearchParams,
  customRangeDescription,
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
      from: "",
      to: "",
      kind: "chat",
      apiKey: "",
      model: "",
      cacheMissOnly: false,
      tagKey: "",
    });
  });

  test("reads filters and rejects unknown range/kind", () => {
    const params = new URLSearchParams(
      "range=7d&kind=embedding&api_key=abc&model=deepseek-v4-flash&cache_miss_only=true&tag_key=feature",
    );
    expect(parseAnalysisQuery(params)).toEqual({
      range: "7d",
      from: "",
      to: "",
      kind: "embedding",
      apiKey: "abc",
      model: "deepseek-v4-flash",
      cacheMissOnly: true,
      tagKey: "feature",
    });
    expect(
      parseAnalysisQuery(new URLSearchParams("range=year&kind=rerank")),
    ).toEqual({
      range: "24h",
      from: "",
      to: "",
      kind: "chat",
      apiKey: "",
      model: "",
      cacheMissOnly: false,
      tagKey: "",
    });
  });

  test("reads a valid custom range as ISO strings", () => {
    const params = new URLSearchParams(
      "range=custom&from=2026-08-21T05:00:00Z&to=2026-08-23T09:30:00Z",
    );
    expect(parseAnalysisQuery(params)).toMatchObject({
      range: "custom",
      from: "2026-08-21T05:00:00.000Z",
      to: "2026-08-23T09:30:00.000Z",
    });
  });

  test("falls back to 24h for invalid custom ranges", () => {
    for (const qs of [
      "range=custom", // missing from/to
      "range=custom&from=2026-08-23T09:30:00Z", // missing to
      "range=custom&from=2026-08-21T05:00:00Z&to=2026-08-21T05:00:00Z", // to == from
      "range=custom&from=2026-08-23T09:30:00Z&to=2026-08-21T05:00:00Z", // to < from
      "range=custom&from=nonsense&to=2026-08-21T05:00:00Z", // unparseable from
    ]) {
      expect(parseAnalysisQuery(new URLSearchParams(qs))).toMatchObject({
        range: "24h",
        from: "",
        to: "",
      });
    }
  });

  test("ignores stray from/to when range is a preset", () => {
    expect(
      parseAnalysisQuery(
        new URLSearchParams(
          "range=7d&from=2026-08-21T05:00:00Z&to=2026-08-23T09:30:00Z",
        ),
      ),
    ).toMatchObject({
      range: "7d",
      from: "",
      to: "",
    });
  });
});

describe("analysisSearchParams", () => {
  test("omits defaults", () => {
    expect(
      analysisSearchParams({
        range: "24h",
        from: "",
        to: "",
        kind: "chat",
        apiKey: "",
        model: "",
        cacheMissOnly: false,
        tagKey: "",
      }).toString(),
    ).toBe("");
    expect(
      analysisSearchParams({
        range: "15m",
        from: "",
        to: "",
        kind: "embedding",
        apiKey: "o6bTIwfcUBKV",
        model: "text-embedding-3-small",
        cacheMissOnly: true,
        tagKey: "feature",
      }).toString(),
    ).toBe(
      "range=15m&kind=embedding&api_key=o6bTIwfcUBKV&model=text-embedding-3-small&cache_miss_only=true&tag_key=feature",
    );
  });

  test("serializes a custom range", () => {
    expect(
      analysisSearchParams({
        range: "custom",
        from: "2026-08-21T05:00:00.000Z",
        to: "2026-08-23T09:30:00.000Z",
        kind: "chat",
        apiKey: "",
        model: "",
        cacheMissOnly: false,
        tagKey: "",
      }).toString(),
    ).toBe(
      "range=custom&from=2026-08-21T05%3A00%3A00.000Z&to=2026-08-23T09%3A30%3A00.000Z",
    );
  });
});

describe("formatters", () => {
  test("compact counts and range copy", () => {
    expect(formatCompactCount(12)).toBe("12");
    expect(formatCompactCount(1500)).toBe("1.5K");
    expect(formatCompactCount(2_300_000)).toBe("2.3M");
    expect(rangeDescription("24h")).toBe("Last 24 hours");
    expect(rangeDescription("custom")).toBe("Custom range");
    expect(formatInputCacheHitDescription(2500, 10_000)).toBe(
      "2.5K / 10.0K input tokens",
    );
  });

  test("custom range description formats both ends", () => {
    expect(
      customRangeDescription(
        "2026-08-21T05:00:00.000Z",
        "2026-08-23T09:30:00.000Z",
      ),
    ).toMatch(/Aug 2[13].+–.+Aug 2[13]/);
    expect(customRangeDescription("nonsense", "nonsense")).toBe("Custom range");
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

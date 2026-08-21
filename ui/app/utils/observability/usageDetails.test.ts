// Modified by Delta-AI under Apache 2.0
import { describe, expect, test } from "vitest";
import type { ParsedModelInferenceRow } from "~/utils/clickhouse/inference";
import {
  buildInferenceUsageDetails,
  extractProviderUsage,
  formatOutputTps,
  outputTpsExcludingTtft,
} from "./usageDetails";

function makeModelInference(
  overrides: Partial<ParsedModelInferenceRow>,
): ParsedModelInferenceRow {
  return {
    id: "00000000-0000-7000-0000-000000000001",
    inference_id: "00000000-0000-7000-0000-000000000000",
    raw_request: undefined,
    raw_response: undefined,
    model_name: "deepseek-v4-flash",
    model_provider_name: "deepseek",
    response_time_ms: 1687,
    ttft_ms: undefined,
    timestamp: "2026-08-20T11:00:00Z",
    system: undefined,
    input_messages: [],
    output: [],
    cached: false,
    cost: undefined,
    ...overrides,
  };
}

describe("outputTpsExcludingTtft", () => {
  test("excludes TTFT from generation time", () => {
    expect(
      outputTpsExcludingTtft({
        outputTokens: 44,
        durationMs: 1577,
        ttftMs: 200,
      }),
    ).toBeCloseTo((44 * 1000) / 1377, 6);
  });

  test("treats missing TTFT as zero, matching Synapse COALESCE", () => {
    expect(
      outputTpsExcludingTtft({
        outputTokens: 44,
        durationMs: 1606,
      }),
    ).toBeCloseTo((44 * 1000) / 1606, 6);
  });

  test("uses a 1ms floor when generation time is empty", () => {
    expect(
      outputTpsExcludingTtft({
        outputTokens: 10,
        durationMs: 5,
        ttftMs: 5,
      }),
    ).toBe(10000);
  });

  test("returns undefined without tokens or duration", () => {
    expect(
      outputTpsExcludingTtft({ outputTokens: 0, durationMs: 1000 }),
    ).toBeUndefined();
    expect(
      outputTpsExcludingTtft({ outputTokens: 10, durationMs: undefined }),
    ).toBeUndefined();
  });

  test("formats two decimal tok/s", () => {
    expect(formatOutputTps(27.396)).toBe("27.40 tok/s");
  });
});

describe("extractProviderUsage", () => {
  test("reads OpenAI-style usage including nested details", () => {
    const usage = extractProviderUsage(
      JSON.stringify({
        id: "chatcmpl-1",
        usage: {
          prompt_tokens: 90,
          completion_tokens: 25,
          total_tokens: 115,
          prompt_tokens_details: { cached_tokens: 12 },
          completion_tokens_details: { reasoning_tokens: 4 },
        },
      }),
    );
    expect(usage).toEqual({
      prompt_tokens: 90,
      completion_tokens: 25,
      total_tokens: 115,
      prompt_tokens_details: { cached_tokens: 12 },
      completion_tokens_details: { reasoning_tokens: 4 },
    });
  });

  test("reads Anthropic-style usage", () => {
    const usage = extractProviderUsage(
      JSON.stringify({
        usage: {
          input_tokens: 90,
          output_tokens: 25,
          cache_read_input_tokens: 40,
          cache_creation_input_tokens: 8,
        },
      }),
    );
    expect(usage).toEqual({
      input_tokens: 90,
      output_tokens: 25,
      cache_read_input_tokens: 40,
      cache_creation_input_tokens: 8,
    });
  });

  test("reads Gemini usageMetadata", () => {
    const usage = extractProviderUsage(
      JSON.stringify({
        usageMetadata: {
          promptTokenCount: 10,
          candidatesTokenCount: 3,
          totalTokenCount: 13,
        },
      }),
    );
    expect(usage).toEqual({
      promptTokenCount: 10,
      candidatesTokenCount: 3,
      totalTokenCount: 13,
    });
  });

  test("walks the last SSE-style array entry", () => {
    const usage = extractProviderUsage(
      JSON.stringify([
        { delta: "hi" },
        { usage: { prompt_tokens: 2, completion_tokens: 1 } },
      ]),
    );
    expect(usage).toEqual({ prompt_tokens: 2, completion_tokens: 1 });
  });

  test("returns null for missing or non-usage payloads", () => {
    expect(extractProviderUsage(undefined)).toBeNull();
    expect(extractProviderUsage("{not json")).toBeNull();
    expect(extractProviderUsage(JSON.stringify({ id: "x" }))).toBeNull();
  });
});

describe("buildInferenceUsageDetails", () => {
  test("prefers tags when model-inference cost is missing", () => {
    const details = buildInferenceUsageDetails({
      tags: {
        "tensorzero::input_tokens": "90",
        "tensorzero::output_tokens": "25",
        "tensorzero::cost": "0.0042",
        "tensorzero::currency": "CNY",
      },
      processingTimeMs: 1687,
      modelInferences: [
        makeModelInference({
          input_tokens: 90,
          output_tokens: 25,
          provider_cache_read_input_tokens: 12,
          provider_cache_write_input_tokens: 0,
        }),
      ],
    });
    expect(details.rows).toEqual(
      expect.arrayContaining([
        { key: "input_tokens", label: "Input tokens", value: 90 },
        { key: "output_tokens", label: "Output tokens", value: 25 },
        { key: "cache_read_tokens", label: "Cache read tokens", value: 12 },
        { key: "cache_write_tokens", label: "Cache write tokens", value: 0 },
        { key: "cost", label: "Cost", value: "¥0.0042" },
        {
          key: "processing_time_ms",
          label: "Processing time",
          value: "1687 ms",
        },
        { key: "ttft_ms", label: "TTFT", value: "—" },
        {
          key: "output_tps",
          label: "Output tok/s (ex-TTFT)",
          value: "14.82 tok/s",
        },
      ]),
    );
  });

  test("includes TTFT and output speed excluding TTFT", () => {
    const details = buildInferenceUsageDetails({
      processingTimeMs: 1687,
      ttftMs: 210,
      modelInferences: [
        makeModelInference({
          output_tokens: 25,
          response_time_ms: 1662,
          ttft_ms: 210,
        }),
      ],
    });
    expect(details.rows).toEqual(
      expect.arrayContaining([
        { key: "ttft_ms", label: "TTFT", value: "210 ms" },
        {
          key: "output_tps",
          label: "Output tok/s (ex-TTFT)",
          value: "17.22 tok/s",
        },
      ]),
    );
  });

  test("includes provider usage from raw_response", () => {
    const details = buildInferenceUsageDetails({
      modelInferences: [
        makeModelInference({
          input_tokens: 90,
          output_tokens: 25,
          raw_response: JSON.stringify({
            usage: {
              prompt_tokens: 90,
              completion_tokens: 25,
              prompt_cache_hit_tokens: 12,
            },
          }),
        }),
      ],
    });
    expect(details.providerBlocks).toEqual([
      {
        id: "00000000-0000-7000-0000-000000000001",
        title: "Provider usage",
        usage: {
          prompt_tokens: 90,
          completion_tokens: 25,
          prompt_cache_hit_tokens: 12,
        },
      },
    ]);
  });
});

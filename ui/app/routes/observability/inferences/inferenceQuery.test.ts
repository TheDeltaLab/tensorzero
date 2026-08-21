// Modified by Delta-AI under Apache 2.0
import { describe, expect, test } from "vitest";
import {
  API_KEY_PUBLIC_ID_TAG,
  CACHED_TAG,
  COST_TAG,
  CURRENCY_TAG,
  INPUT_TOKENS_TAG,
  OUTPUT_TOKENS_TAG,
  PROVIDER_REQUEST_ID_TAG,
  PROVIDER_TAG,
  SERVED_BY_TAG,
  STATUS_CODE_TAG,
  SYNAPSE_REQUEST_ID_TAG,
  apiKeysForSelect,
  buildDimensionalFilter,
  cachedFromTags,
  combineFilters,
  formatApiKeyOption,
  fromDatetimeLocalValue,
  isUuid,
  modelOptionsForProvider,
  parseInferenceQuery,
  providerFromTags,
  providersFromConfig,
  requestIdFromTags,
  requestIdTagFilter,
  statusCodeFromTags,
  usageFromTags,
} from "./inferenceQuery";

describe("inferenceQuery", () => {
  test("parses dimensional query params", () => {
    const params = new URLSearchParams(
      "request_id=caller-1&provider=dummy&cached=true",
    );
    expect(parseInferenceQuery(params)).toEqual({
      requestId: "caller-1",
      startTime: "",
      endTime: "",
      provider: "dummy",
      apiKey: "",
      cached: "true",
      tags: "",
    });
  });

  test("request id matches synapse or provider tag", () => {
    const filter = buildDimensionalFilter({
      requestId: "caller-1",
      startTime: "",
      endTime: "",
      provider: "",
      apiKey: "",
      cached: "all",
      tags: "",
    });
    expect(filter).toEqual({
      type: "or",
      children: [
        {
          type: "tag",
          key: SYNAPSE_REQUEST_ID_TAG,
          value: "caller-1",
          comparison_operator: "=",
        },
        {
          type: "tag",
          key: PROVIDER_REQUEST_ID_TAG,
          value: "caller-1",
          comparison_operator: "=",
        },
      ],
    });
  });

  test("requestIdTagFilter ORs synapse and provider request ids", () => {
    expect(requestIdTagFilter("abc-123")).toEqual({
      type: "or",
      children: [
        {
          type: "tag",
          key: SYNAPSE_REQUEST_ID_TAG,
          value: "abc-123",
          comparison_operator: "=",
        },
        {
          type: "tag",
          key: PROVIDER_REQUEST_ID_TAG,
          value: "abc-123",
          comparison_operator: "=",
        },
      ],
    });
    expect(requestIdTagFilter("  ")).toBeUndefined();
  });

  test("omits request-id tags when includeRequestIdTags is false", () => {
    const filter = buildDimensionalFilter(
      {
        requestId: "abc-123",
        startTime: "",
        endTime: "",
        provider: "dummy",
        apiKey: "",
        cached: "true",
        tags: "",
      },
      { includeRequestIdTags: false },
    );
    expect(filter).toEqual({
      type: "and",
      children: [
        {
          type: "tag",
          key: PROVIDER_TAG,
          value: "dummy",
          comparison_operator: "=",
        },
        {
          type: "tag",
          key: CACHED_TAG,
          value: "true",
          comparison_operator: "=",
        },
      ],
    });
  });

  test("provider without slash filters tensorzero::provider", () => {
    const filter = buildDimensionalFilter({
      requestId: "",
      startTime: "",
      endTime: "",
      provider: "dummy",
      apiKey: "",
      cached: "all",
      tags: "",
    });
    expect(filter).toEqual({
      type: "tag",
      key: PROVIDER_TAG,
      value: "dummy",
      comparison_operator: "=",
    });
  });

  test("provider with slash filters served_by", () => {
    const filter = buildDimensionalFilter({
      requestId: "",
      startTime: "",
      endTime: "",
      provider: "dummy/good",
      apiKey: "",
      cached: "all",
      tags: "",
    });
    expect(filter).toEqual({
      type: "tag",
      key: SERVED_BY_TAG,
      value: "dummy/good",
      comparison_operator: "=",
    });
  });

  test("combines time range, cache, and request id with AND", () => {
    const start = "2026-08-19T10:00";
    const filter = buildDimensionalFilter({
      requestId: "req-1",
      startTime: start,
      endTime: "",
      provider: "",
      apiKey: "",
      cached: "false",
      tags: "",
    });
    expect(filter?.type).toBe("and");
    if (filter?.type !== "and") return;
    expect(filter.children).toHaveLength(3);
    expect(filter.children[1]).toMatchObject({
      type: "tag",
      key: CACHED_TAG,
      value: "false",
    });
    const startIso = fromDatetimeLocalValue(start);
    expect(startIso).toBeTruthy();
    expect(filter.children).toContainEqual({
      type: "time",
      time: startIso,
      comparison_operator: ">=",
    });
  });

  test("combineFilters ANDs dimensional and advanced filters", () => {
    const combined = combineFilters(
      {
        type: "tag",
        key: PROVIDER_TAG,
        value: "dummy",
        comparison_operator: "=",
      },
      {
        type: "tag",
        key: "env",
        value: "prod",
        comparison_operator: "=",
      },
    );
    expect(combined).toEqual({
      type: "and",
      children: [
        {
          type: "tag",
          key: PROVIDER_TAG,
          value: "dummy",
          comparison_operator: "=",
        },
        {
          type: "tag",
          key: "env",
          value: "prod",
          comparison_operator: "=",
        },
      ],
    });
  });

  test("isUuid and requestIdFromTags helpers", () => {
    expect(isUuid("0196372f-1b4b-7013-a446-511e312a3c30")).toBe(true);
    expect(isUuid("caller-trace-1")).toBe(false);
    expect(
      requestIdFromTags({
        [SYNAPSE_REQUEST_ID_TAG]: "syn-1",
        [PROVIDER_REQUEST_ID_TAG]: "prov-1",
      }),
    ).toBe("syn-1");
  });

  test("providersFromConfig uses configured provider names, not model ids", () => {
    const providers = providersFromConfig({
      model_names: ["deepseek-v4-flash", "dummy-chat"],
      embedding_model_names: ["qwen3-embedding-4b"],
      model_providers: {
        "deepseek-v4-flash": ["synapse"],
        "dummy-chat": ["dummy"],
      },
      embedding_model_providers: {
        "qwen3-embedding-4b": ["synapse"],
      },
      model_aliases: [
        {
          name: "deepseek-v4-flash",
          targets: [{ provider: "synapse", model: "deepseek-v4-flash" }],
        },
        {
          name: "dummy-chat",
          targets: [{ provider: "dummy", model: "good" }],
        },
        {
          name: "qwen3-embedding-4b",
          targets: [{ provider: "synapse", model: "qwen3-embedding-4b" }],
        },
      ],
    });
    expect(providers.map((provider) => provider.id)).toEqual([
      "dummy",
      "synapse",
    ]);
    expect(
      providers.find((provider) => provider.id === "synapse")?.models,
    ).toEqual(["deepseek-v4-flash", "qwen3-embedding-4b"]);
    expect(
      providers.find((provider) => provider.id === "dummy")?.models,
    ).toEqual(["dummy-chat", "dummy::good"]);
  });

  test("providersFromConfig falls back to provider::model prefixes only", () => {
    const providers = providersFromConfig({
      model_names: ["dummy::good", "deepseek-v4-flash", "openai::gpt-4o"],
      embedding_model_names: ["dummy::embedding"],
      model_providers: {},
      embedding_model_providers: {},
      model_aliases: [],
    });
    expect(providers.map((provider) => provider.id)).toEqual([
      "dummy",
      "openai",
    ]);
    expect(providers[0].models).toEqual(["dummy::embedding", "dummy::good"]);
  });

  test("modelOptionsForProvider includes aliases only for all providers", () => {
    const providers = providersFromConfig({
      model_names: ["dummy::good"],
      embedding_model_names: [],
      model_providers: { "dummy::good": ["dummy"] },
      embedding_model_providers: {},
      model_aliases: [],
    });
    expect(modelOptionsForProvider(providers, "", ["qwen"])).toEqual([
      "dummy::good",
      "qwen",
    ]);
    expect(modelOptionsForProvider(providers, "dummy", ["qwen"])).toEqual([
      "dummy::good",
    ]);
  });

  test("tag helpers expose provider, status, cache, and usage", () => {
    expect(
      providerFromTags({ [PROVIDER_TAG]: "dummy" }, "openai::gpt-4o"),
    ).toBe("dummy");
    expect(providerFromTags({}, "openai::gpt-4o")).toBe("openai");
    expect(statusCodeFromTags({})).toBe(200);
    expect(statusCodeFromTags({ [STATUS_CODE_TAG]: "429" })).toBe(429);
    expect(cachedFromTags({ [CACHED_TAG]: "true" })).toBe(true);
    expect(
      usageFromTags({
        [INPUT_TOKENS_TAG]: "10",
        [OUTPUT_TOKENS_TAG]: "4",
        [COST_TAG]: "0.0012",
      }),
    ).toEqual({
      input_tokens: 10,
      output_tokens: 4,
      cost: 0.0012,
      currency: "USD",
    });
    expect(
      usageFromTags({
        [INPUT_TOKENS_TAG]: "10",
        [OUTPUT_TOKENS_TAG]: "4",
        [COST_TAG]: "6",
        [CURRENCY_TAG]: "CNY",
      }),
    ).toEqual({
      input_tokens: 10,
      output_tokens: 4,
      cost: 6,
      currency: "CNY",
    });
  });

  test("user tags from the search bar become AND tag filters", () => {
    const filter = buildDimensionalFilter({
      requestId: "",
      startTime: "",
      endTime: "",
      provider: "",
      apiKey: "",
      cached: "all",
      tags: "env=prod,team=ml",
    });
    expect(filter).toEqual({
      type: "and",
      children: [
        {
          type: "tag",
          key: "env",
          value: "prod",
          comparison_operator: "=",
        },
        {
          type: "tag",
          key: "team",
          value: "ml",
          comparison_operator: "=",
        },
      ],
    });
  });

  test("parses api_key query param", () => {
    const params = new URLSearchParams("api_key=abcdefghijkl");
    expect(parseInferenceQuery(params).apiKey).toBe("abcdefghijkl");
  });

  test("api key filters tensorzero::api_key_public_id", () => {
    const filter = buildDimensionalFilter({
      requestId: "",
      startTime: "",
      endTime: "",
      provider: "",
      apiKey: "abcdefghijkl",
      cached: "all",
      tags: "",
    });
    expect(filter).toEqual({
      type: "tag",
      key: API_KEY_PUBLIC_ID_TAG,
      value: "abcdefghijkl",
      comparison_operator: "=",
    });
  });

  test("apiKeysForSelect keeps a selected id that is not in the list", () => {
    const options = apiKeysForSelect(
      [
        {
          public_id: "abcdefghijkl",
          description: "prod",
        },
      ],
      "missingkey12",
    );
    expect(options.map((key) => key.public_id)).toEqual([
      "missingkey12",
      "abcdefghijkl",
    ]);
    expect(formatApiKeyOption(options[1])).toBe("prod (abcdefghijkl)");
    expect(
      formatApiKeyOption({
        public_id: "zzzzzzzzzzzz",
        description: "old",
        disabled: true,
      }),
    ).toBe("old (zzzzzzzzzzzz) (disabled)");
  });
});

// Modified by Delta-AI under Apache 2.0
import { describe, expect, test } from "vitest";
import type {
  Input,
  StoredChatInference,
  StoredInference,
} from "~/types/tensorzero";
import { EMBEDDING_FUNCTION, RERANK_FUNCTION } from "~/utils/constants";
import {
  embeddingInputTexts,
  formatRelevanceScore,
  inferenceKindFromStored,
  observabilityInferenceKind,
  parseStandaloneOutput,
  rerankInputView,
} from "./standaloneInference";

function chatInference(
  overrides: Partial<Omit<StoredChatInference, "type">> & {
    function_name: string;
  },
): StoredInference {
  return {
    type: "chat",
    variant_name: "dummy::good",
    episode_id: "00000000-0000-0000-0000-000000000001",
    inference_id: "00000000-0000-0000-0000-000000000002",
    timestamp: "2026-08-20T00:00:00Z",
    tags: {},
    dispreferred_outputs: [],
    provider_tools: [],
    ...overrides,
  };
}

describe("observabilityInferenceKind", () => {
  test("uses endpoint tag for already-written default-function rows", () => {
    expect(
      observabilityInferenceKind({
        functionName: "tensorzero::default",
        functionType: "chat",
        tags: { "tensorzero::endpoint": "embeddings" },
      }),
    ).toBe("embedding");
    expect(
      observabilityInferenceKind({
        functionName: "tensorzero::default",
        functionType: "chat",
        tags: { "tensorzero::endpoint": "rerank" },
      }),
    ).toBe("rerank");
  });

  test("uses dedicated function names without tags", () => {
    expect(
      observabilityInferenceKind({ functionName: EMBEDDING_FUNCTION }),
    ).toBe("embedding");
    expect(observabilityInferenceKind({ functionName: RERANK_FUNCTION })).toBe(
      "rerank",
    );
  });
});

describe("parseStandaloneOutput", () => {
  test("parses embedding JSON payload without treating it as chat text", () => {
    const inference = chatInference({
      function_name: EMBEDDING_FUNCTION,
      output: [
        {
          type: "text",
          text: JSON.stringify({
            kind: "embedding",
            count: 2,
            dimensions: 8,
            vectors_omitted: true,
            summary: "Generated 2 embeddings (8 dimensions)",
          }),
        },
      ],
    });
    expect(parseStandaloneOutput(inference, "embedding")).toEqual({
      kind: "embedding",
      count: 2,
      dimensions: 8,
      summary: "Generated 2 embeddings (8 dimensions)",
    });
  });

  test("parses legacy embedding summary text", () => {
    const inference = chatInference({
      function_name: "tensorzero::default",
      tags: { "tensorzero::endpoint": "embeddings" },
      output: [{ type: "text", text: "Generated 1 embedding (3 dimensions)" }],
    });
    expect(inferenceKindFromStored(inference)).toBe("embedding");
    expect(parseStandaloneOutput(inference, "embedding")).toEqual({
      kind: "embedding",
      count: 1,
      dimensions: 3,
      summary: "Generated 1 embedding (3 dimensions)",
    });
  });

  test("parses rerank results without document bodies", () => {
    const inference = chatInference({
      function_name: RERANK_FUNCTION,
      output: [
        {
          type: "text",
          text: JSON.stringify({
            kind: "rerank",
            count: 2,
            results: [
              { index: 1, relevance_score: 0.91 },
              { index: 0, score: 0.2 },
            ],
            summary: "Reranked 2 documents",
          }),
        },
      ],
    });
    expect(parseStandaloneOutput(inference, "rerank")).toEqual({
      kind: "rerank",
      count: 2,
      results: [
        { index: 1, relevanceScore: 0.91 },
        { index: 0, relevanceScore: 0.2 },
      ],
      summary: "Reranked 2 documents",
    });
  });
});

describe("input views", () => {
  test("reads embedding texts from user messages", () => {
    const input: Input = {
      messages: [
        { role: "user", content: [{ type: "text", text: "hello" }] },
        { role: "user", content: [{ type: "text", text: "world" }] },
      ],
    };
    expect(embeddingInputTexts(input)).toEqual(["hello", "world"]);
  });

  test("reads rerank query from system and documents from messages", () => {
    const input: Input = {
      system: "capital",
      messages: [
        { role: "user", content: [{ type: "text", text: "Paris" }] },
        { role: "user", content: [{ type: "text", text: "London" }] },
      ],
    };
    expect(rerankInputView(input)).toEqual({
      query: "capital",
      documents: ["Paris", "London"],
    });
  });

  test("parses legacy Query:/Document prefixes", () => {
    const input: Input = {
      messages: [
        { role: "user", content: [{ type: "text", text: "Query: capital" }] },
        {
          role: "user",
          content: [{ type: "text", text: "Document 0: Paris" }],
        },
      ],
    };
    expect(rerankInputView(input)).toEqual({
      query: "capital",
      documents: ["Paris"],
    });
  });
});

describe("formatRelevanceScore", () => {
  test("trims trailing zeros", () => {
    expect(formatRelevanceScore(0.9)).toBe("0.9");
    expect(formatRelevanceScore(undefined)).toBe("—");
  });
});

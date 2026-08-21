// Modified by Delta-AI under Apache 2.0
import { describe, expect, test } from "vitest";
import {
  extractAssistantReply,
  extractEmbeddings,
  extractRerankResults,
} from "./openai";

describe("extractAssistantReply", () => {
  test("reads string content from chat completions", () => {
    expect(
      extractAssistantReply({
        choices: [{ message: { content: "hello" } }],
      }),
    ).toBe("hello");
  });
});

describe("extractEmbeddings", () => {
  test("reads embedding vectors", () => {
    expect(
      extractEmbeddings({
        data: [{ embedding: [0.1, 0.2], index: 0 }],
      }),
    ).toEqual([{ index: 0, embedding: [0.1, 0.2] }]);
  });
});

describe("extractRerankResults", () => {
  test("falls back to submitted documents when the response omits text", () => {
    expect(
      extractRerankResults({ results: [{ index: 1, relevance_score: 0.9 }] }, [
        "first",
        "second",
      ]),
    ).toEqual([{ index: 1, relevanceScore: 0.9, document: "second" }]);
  });
});

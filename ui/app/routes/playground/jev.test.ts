// Modified by Delta-AI under Apache 2.0
import { describe, expect, test } from "vitest";
import {
  answersFromResponse,
  buildSystemOneRequest,
  emptyQuestion,
  type JevQuestionDraft,
} from "./jev";

function question(overrides: Partial<JevQuestionDraft> = {}): JevQuestionDraft {
  return { ...emptyQuestion(1), id: "refund", ...overrides };
}

describe("buildSystemOneRequest", () => {
  test("builds a noul and parses a JSON object state", () => {
    const built = buildSystemOneRequest({
      model: "jev-1.13",
      state: '{"ticket":"charged twice"}',
      questions: [
        question({
          instructions: "Is a refund requested?",
          trueLabel: "Asks for money back",
          falseLabel: "Does not",
        }),
      ],
    });
    expect(built.ok).toBe(true);
    if (!built.ok) return;
    expect(built.request.state).toEqual({ ticket: "charged twice" });
    expect(built.request.questions.refund).toEqual({
      type: "noul",
      instructions: "Is a refund requested?",
      criteria: { true: "Asks for money back", false: "Does not" },
    });
  });

  test("builds choice and score criteria", () => {
    const built = buildSystemOneRequest({
      model: "jev-latest",
      state: "Help, payouts failed.",
      questions: [
        question({
          id: "team",
          type: "choice",
          instructions: "Which team?",
          options: [
            { key: "billing", description: "Payments" },
            { key: "technical", description: "" },
          ],
        }),
        question({
          id: "urgency",
          type: "score",
          instructions: "How urgent?",
          levels: ["Can wait", "", "Today"],
        }),
      ],
    });
    expect(built.ok).toBe(true);
    if (!built.ok) return;
    expect(built.request.questions.team).toEqual({
      type: "choice",
      instructions: "Which team?",
      criteria: { billing: "Payments", technical: null },
    });
    expect(built.request.questions.urgency).toEqual({
      type: "score",
      instructions: "How urgent?",
      criteria: ["Can wait", "Today"],
    });
  });

  test("rejects a score with one level", () => {
    const built = buildSystemOneRequest({
      model: "jev",
      state: "hello",
      questions: [
        question({
          type: "score",
          instructions: "How urgent?",
          levels: ["Only one", ""],
        }),
      ],
    });
    expect(built).toEqual({
      ok: false,
      error: "Question 1 needs at least two score levels.",
    });
  });
});

describe("answersFromResponse", () => {
  test("summarizes noul, choice, and score", () => {
    expect(
      answersFromResponse({
        answers: {
          refund: { type: "noul", noul: 0.98 },
          team: {
            type: "choice",
            choice: "billing",
            confidence: 0.85,
            probabilities: { billing: 0.9, technical: 0.1 },
          },
          urgency: {
            type: "score",
            score: 1.19,
            confidence: 0.71,
            legend: { "1": "Frustrated", "0": "Calm" },
          },
        },
      }),
    ).toEqual([
      {
        id: "refund",
        type: "noul",
        headline: "0.98",
        lines: ["Probability of yes"],
      },
      {
        id: "team",
        type: "choice",
        headline: "billing",
        lines: ["Confidence 0.85", "billing 0.9", "technical 0.1"],
      },
      {
        id: "urgency",
        type: "score",
        headline: "1.19",
        lines: ["Confidence 0.71", "0. Calm", "1. Frustrated"],
      },
    ]);
  });
});

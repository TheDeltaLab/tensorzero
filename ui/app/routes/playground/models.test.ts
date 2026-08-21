// Modified by Delta-AI under Apache 2.0
import { describe, expect, test } from "vitest";
import {
  aliasesForTask,
  defaultChoice,
  defaultModelName,
  modelsForProvider,
  nonEmptyItems,
  playgroundChoices,
  playgroundRequestModel,
  providersFromChoices,
} from "./models";

describe("aliasesForTask", () => {
  test("keeps wildcard aliases and matching task aliases", () => {
    expect(
      aliasesForTask(
        [
          { name: "deepseek-v4-flash", task: "chat" },
          { name: "qwen3-embedding-4b", task: "embedding" },
          { name: "shared" },
        ],
        "chat",
      ),
    ).toEqual(["deepseek-v4-flash", "shared"]);
  });
});

describe("defaultModelName", () => {
  test("prefers aliases over fallback model names", () => {
    expect(defaultModelName(["alias"], ["model"])).toBe("alias");
    expect(defaultModelName([], ["model"])).toBe("model");
    expect(defaultModelName([], [])).toBe("");
  });
});

describe("nonEmptyItems", () => {
  test("keeps multiline items and drops blank ones", () => {
    expect(nonEmptyItems(["hello\nworld", "  ", "keep"])).toEqual([
      "hello\nworld",
      "keep",
    ]);
  });
});

describe("playgroundChoices", () => {
  test("includes alias targets and configured providers", () => {
    const choices = playgroundChoices(
      [
        {
          name: "deepseek-v4-flash",
          task: "chat",
          targets: [{ provider: "synapse", model: "deepseek-v4-flash" }],
        },
        {
          name: "dummy-chat",
          task: "chat",
          targets: [{ provider: "dummy", model: "good" }],
        },
      ],
      "chat",
      { "deepseek-v4-flash": ["synapse"], "dummy-chat": ["dummy"] },
    );
    expect(providersFromChoices(choices)).toEqual(["dummy", "synapse"]);
    expect(modelsForProvider(choices, "synapse")).toEqual([
      "deepseek-v4-flash",
    ]);
    expect(modelsForProvider(choices, "dummy")).toEqual(["dummy-chat", "good"]);
  });

  test("defaultChoice uses the first chat alias target", () => {
    const aliases = [
      {
        name: "deepseek-v4-flash",
        task: "chat" as const,
        targets: [{ provider: "synapse", model: "deepseek-v4-flash" }],
      },
    ];
    expect(defaultChoice(aliases, "chat", [])).toEqual({
      provider: "synapse",
      model: "deepseek-v4-flash",
    });
  });

  test("playgroundRequestModel prefers configured model names", () => {
    expect(
      playgroundRequestModel("synapse", "deepseek-v4-flash", {
        "deepseek-v4-flash": ["synapse"],
      }),
    ).toBe("deepseek-v4-flash");
    expect(playgroundRequestModel("dummy", "good", {})).toBe("dummy::good");
  });
});

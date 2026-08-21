// Modified by Delta-AI under Apache 2.0
import type { UiModelAlias } from "~/types/tensorzero";

export type PlaygroundTask = "chat" | "embedding" | "rerank";

export type PlaygroundChoice = {
  provider: string;
  model: string;
};

export type ProviderModelMap = { [key in string]?: string[] };

export function aliasesForTask(
  aliases: UiModelAlias[] | undefined,
  task: PlaygroundTask,
): string[] {
  return (aliases ?? [])
    .filter((alias) => alias.task === undefined || alias.task === task)
    .map((alias) => alias.name);
}

export function defaultModelName(
  aliasNames: string[],
  fallbackNames: string[],
): string {
  return aliasNames[0] ?? fallbackNames[0] ?? "";
}

export function nonEmptyItems(items: string[]): string[] {
  return items.filter((item) => item.trim().length > 0);
}

export function playgroundChoices(
  aliases: UiModelAlias[] | undefined,
  task: PlaygroundTask,
  configured: ProviderModelMap | undefined,
): PlaygroundChoice[] {
  const seen = new Set<string>();
  const choices: PlaygroundChoice[] = [];
  const add = (provider: string, model: string) => {
    const trimmedProvider = provider.trim();
    const trimmedModel = model.trim();
    if (!trimmedProvider || !trimmedModel) {
      return;
    }
    const key = `${trimmedProvider}::${trimmedModel}`;
    if (seen.has(key)) {
      return;
    }
    seen.add(key);
    choices.push({ provider: trimmedProvider, model: trimmedModel });
  };

  for (const alias of aliases ?? []) {
    if (alias.task !== undefined && alias.task !== task) {
      continue;
    }
    for (const target of alias.targets ?? []) {
      add(target.provider, target.model);
    }
  }

  if (task !== "rerank") {
    for (const [model, providers] of Object.entries(configured ?? {})) {
      for (const provider of providers ?? []) {
        add(provider, model);
      }
    }
  }

  return choices.sort(
    (left, right) =>
      left.provider.localeCompare(right.provider) ||
      left.model.localeCompare(right.model),
  );
}

export function providersFromChoices(choices: PlaygroundChoice[]): string[] {
  return [...new Set(choices.map((choice) => choice.provider))].sort();
}

export function modelsForProvider(
  choices: PlaygroundChoice[],
  provider: string,
): string[] {
  return choices
    .filter((choice) => choice.provider === provider)
    .map((choice) => choice.model);
}

export function defaultChoice(
  aliases: UiModelAlias[] | undefined,
  task: PlaygroundTask,
  choices: PlaygroundChoice[],
): PlaygroundChoice | undefined {
  for (const alias of aliases ?? []) {
    if (alias.task !== undefined && alias.task !== task) {
      continue;
    }
    const target = alias.targets?.[0];
    if (target?.provider && target.model) {
      return { provider: target.provider, model: target.model };
    }
  }
  return choices[0];
}

export function playgroundRequestModel(
  provider: string,
  model: string,
  configured: ProviderModelMap | undefined,
): string {
  if ((configured?.[model] ?? []).includes(provider)) {
    return model;
  }
  return `${provider}::${model}`;
}

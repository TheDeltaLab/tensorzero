// Modified by Delta-AI under Apache 2.0
import type {
  ContentBlockChatOutput,
  Input,
  StoredInference,
  StoredInput,
  System,
} from "~/types/tensorzero";
import {
  DEFAULT_FUNCTION,
  EMBEDDING_FUNCTION,
  RERANK_FUNCTION,
} from "~/utils/constants";

export const ENDPOINT_TAG = "tensorzero::endpoint";

export type ObservabilityInferenceKind =
  | "chat"
  | "json"
  | "embedding"
  | "rerank";

type InferenceInput = Input | StoredInput;

export function observabilityInferenceKind(args: {
  functionName?: string;
  functionType?: string;
  tags?: Record<string, string | undefined>;
}): ObservabilityInferenceKind {
  const endpoint = args.tags?.[ENDPOINT_TAG];
  if (endpoint === "embeddings" || args.functionName === EMBEDDING_FUNCTION) {
    return "embedding";
  }
  if (endpoint === "rerank" || args.functionName === RERANK_FUNCTION) {
    return "rerank";
  }
  if (args.functionType === "json") {
    return "json";
  }
  return "chat";
}

export function inferenceKindFromStored(
  inference: StoredInference,
): ObservabilityInferenceKind {
  return observabilityInferenceKind({
    functionName: inference.function_name,
    functionType: inference.type,
    tags: inference.tags,
  });
}

export function isStandaloneInferenceKind(
  kind: ObservabilityInferenceKind,
): boolean {
  return kind === "embedding" || kind === "rerank";
}

export function isStandaloneFunctionName(functionName: string): boolean {
  return (
    functionName === EMBEDDING_FUNCTION || functionName === RERANK_FUNCTION
  );
}

export function variantTypeForKind(
  kind: ObservabilityInferenceKind,
  functionName: string,
  configuredType?: string,
): string {
  if (kind === "embedding") return "embedding";
  if (kind === "rerank") return "rerank";
  if (configuredType) return configuredType;
  return functionName === DEFAULT_FUNCTION ? "chat_completion" : "unknown";
}

export type EmbeddingOutputView = {
  kind: "embedding";
  count: number;
  dimensions: number;
  summary: string;
};

export type RerankResultView = {
  index: number;
  relevanceScore?: number;
};

export type RerankOutputView = {
  kind: "rerank";
  count: number;
  results: RerankResultView[];
  summary: string;
};

export type StandaloneOutputView = EmbeddingOutputView | RerankOutputView;

function firstOutputText(
  output: StoredInference["output"] | undefined,
): string | undefined {
  if (!Array.isArray(output)) return undefined;
  const block = output.find(
    (item): item is Extract<ContentBlockChatOutput, { type: "text" }> =>
      item.type === "text",
  );
  return block?.text;
}

function parseJsonObject(text: string): Record<string, unknown> | undefined {
  const trimmed = text.trim();
  if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) {
    return undefined;
  }
  try {
    const parsed: unknown = JSON.parse(trimmed);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    return undefined;
  }
  return undefined;
}

function asNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value)
    ? value
    : undefined;
}

function asString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function parseRerankResults(value: unknown): RerankResultView[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) return [];
    const record = item as Record<string, unknown>;
    const index = asNumber(record.index);
    if (index === undefined) return [];
    return [
      {
        index,
        relevanceScore:
          asNumber(record.relevance_score) ?? asNumber(record.score),
      },
    ];
  });
}

const EMBEDDING_SUMMARY_RE =
  /^Generated (\d+) embeddings? \((\d+) dimensions\)$/;
const RERANK_SUMMARY_RE = /^Reranked (\d+) documents$/;

export function parseStandaloneOutput(
  inference: StoredInference,
  kind: ObservabilityInferenceKind,
): StandaloneOutputView | undefined {
  const text = firstOutputText(inference.output);
  if (!text) return undefined;
  const json = parseJsonObject(text);

  if (kind === "embedding") {
    if (json?.kind === "embedding") {
      return {
        kind: "embedding",
        count: asNumber(json.count) ?? 0,
        dimensions: asNumber(json.dimensions) ?? 0,
        summary: asString(json.summary) ?? text,
      };
    }
    const match = text.match(EMBEDDING_SUMMARY_RE);
    if (match) {
      return {
        kind: "embedding",
        count: Number(match[1]),
        dimensions: Number(match[2]),
        summary: text,
      };
    }
    return {
      kind: "embedding",
      count: 0,
      dimensions: 0,
      summary: text,
    };
  }

  if (kind === "rerank") {
    if (json?.kind === "rerank") {
      const results = parseRerankResults(json.results);
      return {
        kind: "rerank",
        count: asNumber(json.count) ?? results.length,
        results,
        summary: asString(json.summary) ?? text,
      };
    }
    const match = text.match(RERANK_SUMMARY_RE);
    return {
      kind: "rerank",
      count: match ? Number(match[1]) : 0,
      results: [],
      summary: text,
    };
  }

  return undefined;
}

function contentText(content: {
  type: string;
  text?: string;
  value?: string;
}): string | undefined {
  if (content.type === "text" && typeof content.text === "string") {
    return content.text;
  }
  if (content.type === "raw_text" && typeof content.value === "string") {
    return content.value;
  }
  return undefined;
}

function messageTexts(input?: InferenceInput): string[] {
  if (!input) return [];
  return input.messages.flatMap((message) =>
    message.content.flatMap((block) => {
      const text = contentText(block);
      return text === undefined ? [] : [text];
    }),
  );
}

function systemText(system?: System): string | undefined {
  return typeof system === "string" ? system : undefined;
}

const QUERY_PREFIX = /^Query:\s*/;
const DOCUMENT_PREFIX = /^Document \d+:\s*/;

export function embeddingInputTexts(input?: InferenceInput): string[] {
  return messageTexts(input);
}

export function rerankInputView(input?: InferenceInput): {
  query: string;
  documents: string[];
} {
  const system = systemText(input?.system);
  const texts = messageTexts(input);
  if (system !== undefined) {
    return { query: system, documents: texts };
  }
  if (texts.length > 0 && QUERY_PREFIX.test(texts[0])) {
    return {
      query: texts[0].replace(QUERY_PREFIX, ""),
      documents: texts
        .slice(1)
        .map((text) => text.replace(DOCUMENT_PREFIX, "")),
    };
  }
  return { query: texts[0] ?? "", documents: texts.slice(1) };
}

export function formatRelevanceScore(score: number | undefined): string {
  if (score === undefined) return "—";
  return score.toFixed(4).replace(/\.?0+$/, "") || "0";
}

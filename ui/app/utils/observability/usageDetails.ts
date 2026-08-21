// Modified by Delta-AI under Apache 2.0
import { formatCost } from "~/utils/cost";
import { getTotalInferenceUsage } from "~/utils/clickhouse/helpers";
import type { ParsedModelInferenceRow } from "~/utils/clickhouse/inference";
import {
  cachedFromTags,
  mergeUsage,
  usageFromTags,
} from "~/routes/observability/inferences/inferenceQuery";

export type UsageDetailRow = {
  key: string;
  label: string;
  value: unknown;
};

export type ProviderUsageBlock = {
  id: string;
  title: string;
  usage: Record<string, unknown>;
};

export type InferenceUsageDetailsModel = {
  rows: UsageDetailRow[];
  providerBlocks: ProviderUsageBlock[];
};

const USAGE_OBJECT_KEYS = [
  "prompt_tokens",
  "completion_tokens",
  "total_tokens",
  "input_tokens",
  "output_tokens",
  "promptTokenCount",
  "candidatesTokenCount",
  "totalTokenCount",
  "cache_read_input_tokens",
  "cache_creation_input_tokens",
  "cache_creation",
  "cached_tokens",
  "reasoning_tokens",
  "prompt_tokens_details",
  "completion_tokens_details",
  "prompt_cache_hit_tokens",
  "prompt_cache_miss_tokens",
];

export function toFiniteMs(
  value: number | bigint | string | null | undefined,
): number | undefined {
  if (value === null || value === undefined || value === "") {
    return undefined;
  }
  if (typeof value === "bigint") {
    const asNumber = Number(value);
    return Number.isFinite(asNumber) ? asNumber : undefined;
  }
  if (typeof value === "number") {
    return Number.isFinite(value) ? value : undefined;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

/**
 * Output tokens per second excluding TTFT, matching Synapse analytics:
 * `output_tokens * 1000 / max(duration_ms - coalesce(ttft_ms, 0), 1)`.
 */
export function outputTpsExcludingTtft(args: {
  outputTokens?: number | null;
  durationMs?: number | bigint | string | null;
  ttftMs?: number | bigint | string | null;
}): number | undefined {
  const outputTokens = args.outputTokens;
  const durationMs = toFiniteMs(args.durationMs);
  if (outputTokens == null || outputTokens <= 0 || durationMs == null) {
    return undefined;
  }
  const ttftMs = toFiniteMs(args.ttftMs) ?? 0;
  const generationMs = Math.max(durationMs - ttftMs, 1);
  return (outputTokens * 1000) / generationMs;
}

export function formatOutputTps(tps: number): string {
  return `${tps.toFixed(2)} tok/s`;
}

export function formatTtftMs(ttftMs?: number | bigint | string | null): string {
  const ms = toFiniteMs(ttftMs);
  return ms == null ? "—" : `${ms} ms`;
}

function sumOptionalTokens(
  values: Array<number | undefined | null>,
): number | undefined {
  let total = 0;
  let any = false;
  for (const value of values) {
    if (value != null) {
      total += value;
      any = true;
    }
  }
  return any ? total : undefined;
}

export function firstFiniteMs(
  values: Array<number | bigint | string | null | undefined>,
): number | undefined {
  for (const value of values) {
    const ms = toFiniteMs(value);
    if (ms != null) {
      return ms;
    }
  }
  return undefined;
}

function looksLikeUsage(value: unknown): boolean {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return false;
  }
  return USAGE_OBJECT_KEYS.some((key) => key in value);
}

function asUsageRecord(value: unknown): Record<string, unknown> | null {
  if (!looksLikeUsage(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

function usageObjectFromUnknown(
  value: unknown,
): Record<string, unknown> | null {
  if (typeof value === "string") {
    try {
      return usageObjectFromUnknown(JSON.parse(value));
    } catch {
      return null;
    }
  }
  if (!value || typeof value !== "object") {
    return null;
  }
  if (Array.isArray(value)) {
    for (let i = value.length - 1; i >= 0; i--) {
      const found = usageObjectFromUnknown(value[i]);
      if (found) {
        return found;
      }
    }
    return null;
  }
  const obj = value as Record<string, unknown>;
  return (
    asUsageRecord(obj.usage) ??
    asUsageRecord(obj.usageMetadata) ??
    asUsageRecord(obj) ??
    usageObjectFromUnknown(obj.data)
  );
}

export function extractProviderUsage(
  rawResponse: string | null | undefined,
): Record<string, unknown> | null {
  if (!rawResponse) {
    return null;
  }
  try {
    return usageObjectFromUnknown(JSON.parse(rawResponse));
  } catch {
    return null;
  }
}

export function usageRecordToRows(
  usage: Record<string, unknown>,
): UsageDetailRow[] {
  return Object.entries(usage).map(([key, value]) => ({
    key,
    label: key,
    value,
  }));
}

function cacheStatus(
  modelInferences: ParsedModelInferenceRow[],
  tags: Record<string, string> | undefined,
): "none" | "full" | "partial" {
  if (modelInferences.length > 0) {
    const cachedCount = modelInferences.filter((row) => row.cached).length;
    if (cachedCount === 0) {
      return "none";
    }
    return cachedCount === modelInferences.length ? "full" : "partial";
  }
  return cachedFromTags(tags) ? "full" : "none";
}

export function buildInferenceUsageDetails(args: {
  tags?: Record<string, string>;
  processingTimeMs?: number | bigint | string | null;
  ttftMs?: number | bigint | string | null;
  modelInferences: ParsedModelInferenceRow[];
}): InferenceUsageDetailsModel {
  const { tags, modelInferences } = args;
  const fromModels =
    modelInferences.length > 0
      ? getTotalInferenceUsage(modelInferences)
      : undefined;
  const usage = mergeUsage(usageFromTags(tags), fromModels);
  const cacheRead = sumOptionalTokens(
    modelInferences.map((row) => row.provider_cache_read_input_tokens),
  );
  const cacheWrite = sumOptionalTokens(
    modelInferences.map((row) => row.provider_cache_write_input_tokens),
  );
  const responseTimeMs = firstFiniteMs(
    modelInferences.map((row) => row.response_time_ms),
  );
  const ttftMs = firstFiniteMs([
    args.ttftMs,
    ...modelInferences.map((row) => row.ttft_ms),
  ]);
  const processingTimeMs = toFiniteMs(args.processingTimeMs);
  const durationMs = responseTimeMs ?? processingTimeMs;
  const outputTps = outputTpsExcludingTtft({
    outputTokens: usage.output_tokens,
    durationMs,
    ttftMs,
  });
  const cached = cacheStatus(modelInferences, tags);

  const rows: UsageDetailRow[] = [];
  const push = (key: string, label: string, value: unknown) => {
    if (value === undefined || value === null || value === "") {
      return;
    }
    rows.push({ key, label, value });
  };

  push("input_tokens", "Input tokens", usage.input_tokens);
  push("output_tokens", "Output tokens", usage.output_tokens);
  push("cache_read_tokens", "Cache read tokens", cacheRead);
  push("cache_write_tokens", "Cache write tokens", cacheWrite);
  if (usage.cost != null) {
    push("cost", "Cost", formatCost(usage.cost, usage.currency ?? undefined));
  }
  if (processingTimeMs != null) {
    push("processing_time_ms", "Processing time", `${processingTimeMs} ms`);
  }
  if (responseTimeMs != null && responseTimeMs !== processingTimeMs) {
    push("response_time_ms", "Response time", `${responseTimeMs} ms`);
  }
  push("ttft_ms", "TTFT", formatTtftMs(ttftMs));
  push(
    "output_tps",
    "Output tok/s (ex-TTFT)",
    outputTps == null ? "—" : formatOutputTps(outputTps),
  );
  if (cached !== "none") {
    push("cached", "Cached", cached === "full" ? "Yes" : "Partial");
  }

  const providerBlocks: ProviderUsageBlock[] = [];
  for (const row of modelInferences) {
    const providerUsage = extractProviderUsage(row.raw_response);
    if (!providerUsage || Object.keys(providerUsage).length === 0) {
      continue;
    }
    const title =
      modelInferences.length > 1
        ? `${row.model_provider_name} / ${row.model_name}`
        : "Provider usage";
    providerBlocks.push({
      id: row.id,
      title,
      usage: providerUsage,
    });
  }

  return { rows, providerBlocks };
}

export function buildModelInferenceUsageDetails(
  inference: ParsedModelInferenceRow,
): InferenceUsageDetailsModel {
  return buildInferenceUsageDetails({
    ttftMs: inference.ttft_ms,
    modelInferences: [inference],
  });
}

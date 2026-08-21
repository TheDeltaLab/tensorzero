// Modified by Delta-AI under Apache 2.0
export const ANALYSIS_RANGES = ["15m", "1h", "24h", "7d", "30d"] as const;
export type AnalysisRange = (typeof ANALYSIS_RANGES)[number];

export const ANALYSIS_KINDS = ["chat", "embedding"] as const;
export type AnalysisKind = (typeof ANALYSIS_KINDS)[number];

export type AnalysisQueryValues = {
  range: AnalysisRange;
  kind: AnalysisKind;
  apiKey: string;
  model: string;
  cacheMissOnly: boolean;
};

export const DEFAULT_ANALYSIS_QUERY: AnalysisQueryValues = {
  range: "24h",
  kind: "chat",
  apiKey: "",
  model: "",
  cacheMissOnly: false,
};

export type AnalysisProviderStats = {
  provider: string;
  count: number;
  percentage: number;
};

export type AnalysisModelStats = {
  model: string;
  provider: string;
  count: number;
  avg_latency: number | null;
  p50: number | null;
  p90: number | null;
  p99: number | null;
};

export type AnalysisCountPoint = {
  date: string;
  count: number;
};

export type AnalysisPercentilePoint = {
  date: string;
  p50: number | null;
  p90: number | null;
  p99?: number | null;
  avg: number | null;
};

export type AnalysisTokenPoint = {
  date: string;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  count: number;
};

export type AnalysisResponse = {
  total_requests: number;
  total_responses: number;
  success_rate: number;
  cache_hit_rate: number;
  input_cache_hit_rate: number;
  cache_read_input_tokens: number;
  unique_providers: number;
  unique_models: number;
  avg_latency: number | null;
  total_tokens: number;
  total_input_tokens: number;
  total_output_tokens: number;
  provider_stats: AnalysisProviderStats[];
  model_stats: AnalysisModelStats[];
  model_latency_stats: AnalysisModelStats[];
  token_usage_over_time: AnalysisTokenPoint[];
  requests_over_time: AnalysisCountPoint[];
  latency_over_time: AnalysisPercentilePoint[];
  ttft_over_time: AnalysisPercentilePoint[];
  output_tps_over_time: AnalysisPercentilePoint[];
};

export function parseAnalysisQuery(
  params: URLSearchParams,
): AnalysisQueryValues {
  const rangeRaw = params.get("range") ?? DEFAULT_ANALYSIS_QUERY.range;
  const kindRaw = params.get("kind") ?? DEFAULT_ANALYSIS_QUERY.kind;
  const cacheMiss = params.get("cache_miss_only");
  return {
    range: ANALYSIS_RANGES.includes(rangeRaw as AnalysisRange)
      ? (rangeRaw as AnalysisRange)
      : DEFAULT_ANALYSIS_QUERY.range,
    kind: ANALYSIS_KINDS.includes(kindRaw as AnalysisKind)
      ? (kindRaw as AnalysisKind)
      : DEFAULT_ANALYSIS_QUERY.kind,
    apiKey: params.get("api_key") ?? "",
    model: params.get("model") ?? "",
    cacheMissOnly: cacheMiss === "true",
  };
}

export function analysisSearchParams(
  query: AnalysisQueryValues,
): URLSearchParams {
  const params = new URLSearchParams();
  if (query.range !== DEFAULT_ANALYSIS_QUERY.range) {
    params.set("range", query.range);
  }
  if (query.kind !== DEFAULT_ANALYSIS_QUERY.kind) {
    params.set("kind", query.kind);
  }
  if (query.apiKey.trim()) {
    params.set("api_key", query.apiKey.trim());
  }
  if (query.model.trim()) {
    params.set("model", query.model.trim());
  }
  if (query.cacheMissOnly) {
    params.set("cache_miss_only", "true");
  }
  return params;
}

export function rangeDescription(range: AnalysisRange): string {
  switch (range) {
    case "15m":
      return "Last 15 minutes";
    case "1h":
      return "Last hour";
    case "24h":
      return "Last 24 hours";
    case "7d":
      return "Last 7 days";
    case "30d":
      return "Last 30 days";
  }
}

export function formatInputCacheHitDescription(
  cacheReadInputTokens: number,
  totalInputTokens: number,
): string {
  return `${formatCompactCount(cacheReadInputTokens)} / ${formatCompactCount(totalInputTokens)} input tokens`;
}

export function formatCompactCount(num: number): string {
  if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(1)}M`;
  if (num >= 1_000) return `${(num / 1_000).toFixed(1)}K`;
  return num.toLocaleString();
}

export function formatBucketLabel(dateStr: string): string {
  const date = new Date(dateStr);
  if (Number.isNaN(date.getTime())) {
    return dateStr;
  }
  if (dateStr.length === 20 && dateStr.includes(":00Z")) {
    return date.toLocaleString("en-US", {
      hour: "numeric",
      minute: "2-digit",
      hour12: true,
    });
  }
  if (dateStr.includes("T")) {
    return date.toLocaleString("en-US", { hour: "numeric", hour12: true });
  }
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

export function analysisModelsForKind(
  kind: AnalysisKind,
  config: {
    model_names: string[];
    embedding_model_names: string[];
    model_aliases: Array<{ name: string }>;
  },
): string[] {
  if (kind === "embedding") {
    return [...config.embedding_model_names].sort();
  }
  return [
    ...new Set([
      ...config.model_aliases.map((alias) => alias.name),
      ...config.model_names,
    ]),
  ].sort();
}

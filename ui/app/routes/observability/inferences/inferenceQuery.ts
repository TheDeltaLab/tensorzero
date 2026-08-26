// Modified by Delta-AI under Apache 2.0
import type {
  InferenceFilter,
  ModelInference,
  UiConfig,
} from "~/types/tensorzero";
import { parseCsvTags } from "~/utils/observability/inferenceTags";

export const SYNAPSE_REQUEST_ID_TAG = "tensorzero::synapse_request_id";
export const PROVIDER_REQUEST_ID_TAG = "tensorzero::provider_request_id";
export const PROVIDER_TAG = "tensorzero::provider";
export const SERVED_BY_TAG = "tensorzero::served_by";
export const CACHED_TAG = "tensorzero::cached";
export const FALLBACK_COUNT_TAG = "tensorzero::fallback_count";
export const INPUT_TOKENS_TAG = "tensorzero::input_tokens";
export const OUTPUT_TOKENS_TAG = "tensorzero::output_tokens";
export const COST_TAG = "tensorzero::cost";
export const CURRENCY_TAG = "tensorzero::currency";
export const STATUS_CODE_TAG = "tensorzero::status_code";
export const API_KEY_PUBLIC_ID_TAG = "tensorzero::api_key_public_id";

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export type InferenceQueryValues = {
  requestId: string;
  startTime: string;
  endTime: string;
  provider: string;
  apiKey: string;
  cached: "all" | "true" | "false";
  tags: string;
};

export const DEFAULT_INFERENCE_QUERY: InferenceQueryValues = {
  requestId: "",
  startTime: "",
  endTime: "",
  provider: "",
  apiKey: "",
  cached: "all",
  tags: "",
};

export type ApiKeySelectOption = {
  public_id: string;
  description?: string | null;
  disabled?: boolean;
};

export type InferenceProviderOption = {
  id: string;
  name: string;
  models: string[];
};

export type InferenceUsageSummary = {
  input_tokens?: number | null;
  output_tokens?: number | null;
  provider_cache_read_input_tokens?: number | null;
  provider_cache_write_input_tokens?: number | null;
  cost?: number | null;
  currency?: string | null;
};

export function isUuid(value: string): boolean {
  return UUID_RE.test(value.trim());
}

export function parseInferenceQuery(
  params: URLSearchParams,
): InferenceQueryValues {
  const cached = params.get("cached");
  return {
    requestId: params.get("request_id") ?? "",
    startTime: params.get("start_time") ?? "",
    endTime: params.get("end_time") ?? "",
    provider: params.get("provider") ?? "",
    apiKey: params.get("api_key") ?? "",
    cached: cached === "true" || cached === "false" ? cached : "all",
    tags: params.get("tags") ?? "",
  };
}

export function toDatetimeLocalValue(iso: string): string {
  if (!iso) return "";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  const pad = (value: number) => String(value).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

export function fromDatetimeLocalValue(value: string): string | undefined {
  if (!value) return undefined;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return undefined;
  return date.toISOString();
}

function tagEquals(key: string, value: string): InferenceFilter {
  return {
    type: "tag",
    key,
    value,
    comparison_operator: "=",
  };
}

export function requestIdTagFilter(
  requestId: string,
): InferenceFilter | undefined {
  const value = requestId.trim();
  if (!value) {
    return undefined;
  }
  return {
    type: "or",
    children: [
      tagEquals(SYNAPSE_REQUEST_ID_TAG, value),
      tagEquals(PROVIDER_REQUEST_ID_TAG, value),
    ],
  };
}

export function buildDimensionalFilter(
  query: InferenceQueryValues,
  options: { includeRequestIdTags?: boolean } = {},
): InferenceFilter | undefined {
  const includeRequestIdTags = options.includeRequestIdTags ?? true;
  const children: InferenceFilter[] = [];
  if (includeRequestIdTags) {
    const requestIdFilter = requestIdTagFilter(query.requestId);
    if (requestIdFilter) {
      children.push(requestIdFilter);
    }
  }
  if (query.provider.trim()) {
    const provider = query.provider.trim();
    children.push(
      provider.includes("/")
        ? tagEquals(SERVED_BY_TAG, provider)
        : tagEquals(PROVIDER_TAG, provider),
    );
  }
  if (query.apiKey.trim()) {
    children.push(tagEquals(API_KEY_PUBLIC_ID_TAG, query.apiKey.trim()));
  }
  if (query.cached !== "all") {
    children.push(tagEquals(CACHED_TAG, query.cached));
  }
  const userTags = parseCsvTags(query.tags);
  for (const [key, value] of Object.entries(userTags)) {
    children.push(tagEquals(key, value));
  }
  const startTime = fromDatetimeLocalValue(query.startTime);
  if (startTime) {
    children.push({
      type: "time",
      time: startTime as unknown as Date,
      comparison_operator: ">=",
    });
  }
  const endTime = fromDatetimeLocalValue(query.endTime);
  if (endTime) {
    children.push({
      type: "time",
      time: endTime as unknown as Date,
      comparison_operator: "<=",
    });
  }
  if (children.length === 0) return undefined;
  if (children.length === 1) return children[0];
  return { type: "and", children };
}

export function combineFilters(
  dimensional?: InferenceFilter,
  advanced?: InferenceFilter,
): InferenceFilter | undefined {
  if (dimensional && advanced) {
    return { type: "and", children: [dimensional, advanced] };
  }
  return dimensional ?? advanced;
}

export function requestIdFromTags(
  tags: Record<string, string> | undefined,
): string | undefined {
  if (!tags) return undefined;
  return tags[SYNAPSE_REQUEST_ID_TAG] ?? tags[PROVIDER_REQUEST_ID_TAG];
}

export function servedByFromTags(
  tags: Record<string, string> | undefined,
): string | undefined {
  if (!tags) return undefined;
  return tags[SERVED_BY_TAG] ?? tags[PROVIDER_TAG];
}

export function providerFromTags(
  tags: Record<string, string> | undefined,
  variantName?: string,
): string | undefined {
  if (tags?.[PROVIDER_TAG]) return tags[PROVIDER_TAG];
  const servedBy = tags?.[SERVED_BY_TAG];
  if (servedBy) {
    return servedBy.split("/")[0] || servedBy;
  }
  if (variantName?.includes("::")) {
    return variantName.slice(0, variantName.indexOf("::"));
  }
  return undefined;
}

export function cachedFromTags(
  tags: Record<string, string> | undefined,
): boolean {
  return tags?.[CACHED_TAG] === "true";
}

export function fallbackCountFromTags(
  tags: Record<string, string> | undefined,
): number {
  const raw = tags?.[FALLBACK_COUNT_TAG];
  if (!raw) return 0;
  const count = Number(raw);
  return Number.isFinite(count) && count > 0 ? count : 0;
}

export function statusCodeFromTags(
  tags: Record<string, string> | undefined,
): number {
  const raw = tags?.[STATUS_CODE_TAG];
  if (raw) {
    const code = Number(raw);
    if (Number.isFinite(code)) return code;
  }
  // Persisted inferences are successful writes.
  return 200;
}

export function usageFromTags(
  tags: Record<string, string> | undefined,
): InferenceUsageSummary {
  if (!tags) return {};
  const cost = parseOptionalNumber(tags[COST_TAG]) ?? null;
  return {
    input_tokens: parseOptionalNumber(tags[INPUT_TOKENS_TAG]),
    output_tokens: parseOptionalNumber(tags[OUTPUT_TOKENS_TAG]),
    cost,
    currency:
      cost != null ? normalizeTagCurrency(tags[CURRENCY_TAG]) : undefined,
  };
}

export function usageFromModelInferences(
  rows: ModelInference[],
): InferenceUsageSummary {
  if (rows.length === 0) return {};
  const currencies = rows.map((row) => normalizeTagCurrency(row.currency));
  const allCostsPresent = rows.every((row) => row.cost != null);
  const sameCurrency = currencies.every((code) => code === currencies[0]);
  const canSumCost = allCostsPresent && sameCurrency;
  return {
    input_tokens: rows.reduce((sum, row) => sum + (row.input_tokens ?? 0), 0),
    output_tokens: rows.reduce((sum, row) => sum + (row.output_tokens ?? 0), 0),
    cost: canSumCost
      ? rows.reduce((sum, row) => sum + (row.cost ?? 0), 0)
      : null,
    currency: canSumCost ? currencies[0] : undefined,
  };
}

export function mergeUsage(
  fromTags: InferenceUsageSummary,
  fromModels?: InferenceUsageSummary,
): InferenceUsageSummary {
  const cost = fromTags.cost ?? fromModels?.cost ?? null;
  return {
    input_tokens: fromTags.input_tokens ?? fromModels?.input_tokens,
    output_tokens: fromTags.output_tokens ?? fromModels?.output_tokens,
    cost,
    currency:
      cost != null
        ? (fromTags.currency ?? fromModels?.currency ?? "USD")
        : undefined,
  };
}

export function toFiniteNumber(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "bigint") return Number(value);
  if (typeof value === "string" && value !== "") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
}

export function providersFromConfig(
  config: Pick<
    UiConfig,
    | "model_names"
    | "embedding_model_names"
    | "model_providers"
    | "embedding_model_providers"
    | "model_aliases"
  >,
): InferenceProviderOption[] {
  const modelsByProvider = new Map<string, Set<string>>();
  const add = (provider: string | undefined, model: string | undefined) => {
    const trimmedProvider = provider?.trim() ?? "";
    const trimmedModel = model?.trim() ?? "";
    if (!trimmedProvider || !trimmedModel) {
      return;
    }
    const models = modelsByProvider.get(trimmedProvider) ?? new Set<string>();
    models.add(trimmedModel);
    modelsByProvider.set(trimmedProvider, models);
  };

  const addFromMap = (
    map: Record<string, string[] | undefined> | undefined,
  ) => {
    for (const [model, providers] of Object.entries(map ?? {})) {
      for (const provider of providers ?? []) {
        add(provider, model);
      }
    }
  };

  addFromMap(config.model_providers);
  addFromMap(config.embedding_model_providers);

  for (const alias of config.model_aliases ?? []) {
    for (const target of alias.targets ?? []) {
      add(target.provider, alias.name);
      if (target.model && target.model !== alias.name) {
        add(target.provider, `${target.provider}::${target.model}`);
      }
    }
  }

  // Older configs (or tests) may only have `provider::model` names and no maps.
  // Never treat an unprefixed model name as a provider.
  if (modelsByProvider.size === 0) {
    for (const name of [
      ...config.model_names,
      ...config.embedding_model_names,
    ]) {
      const separator = name.indexOf("::");
      if (separator > 0) {
        add(name.slice(0, separator), name);
      }
    }
  }

  return [...modelsByProvider.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([id, models]) => ({
      id,
      name: id,
      models: [...models].sort(),
    }));
}

export function modelOptionsForProvider(
  providers: InferenceProviderOption[],
  provider: string,
  aliases: string[],
): string[] {
  const scoped =
    provider.length > 0
      ? (providers.find((item) => item.id === provider)?.models ?? [])
      : [...aliases, ...providers.flatMap((item) => item.models)];
  return [...new Set(scoped)].sort();
}

export function apiKeysForSelect(
  keys: Array<{
    public_id: string;
    description?: string | null;
    disabled?: boolean;
    disabled_at?: string;
  }>,
  selected: string,
): ApiKeySelectOption[] {
  const options: ApiKeySelectOption[] = keys.map((key) => ({
    public_id: key.public_id,
    description: key.description ?? undefined,
    disabled: key.disabled ?? Boolean(key.disabled_at),
  }));
  const trimmed = selected.trim();
  if (trimmed && !options.some((key) => key.public_id === trimmed)) {
    options.unshift({ public_id: trimmed });
  }
  return options;
}

export function formatApiKeyOption(key: ApiKeySelectOption): string {
  const name = key.description?.trim();
  const label = name ? `${name} (${key.public_id})` : key.public_id;
  return key.disabled ? `${label} (disabled)` : label;
}

function parseOptionalNumber(value: string | undefined): number | undefined {
  if (value === undefined || value === "") return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function normalizeTagCurrency(value: string | undefined | null): string {
  const code = (value ?? "USD").trim().toUpperCase();
  if (!code) return "USD";
  if (code === "RMB") return "CNY";
  return code;
}

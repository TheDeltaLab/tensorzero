// Modified by Delta-AI under Apache 2.0
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useNavigate, useSearchParams } from "react-router";
import { Button } from "~/components/ui/button";
import { Input } from "~/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";
import { FunctionSelector } from "~/components/function/FunctionSelector";
import { useAllFunctionConfigs, useConfig } from "~/context/config";
import { X } from "lucide-react";
import {
  DEFAULT_INFERENCE_QUERY,
  apiKeysForSelect,
  formatApiKeyOption,
  fromDatetimeLocalValue,
  modelOptionsForProvider,
  parseInferenceQuery,
  providersFromConfig,
  toDatetimeLocalValue,
  type InferenceQueryValues,
} from "./inferenceQuery";
import type { KeyInfo } from "~/types/tensorzero";

const FILTER_PARAM_KEYS = [
  "request_id",
  "start_time",
  "end_time",
  "provider",
  "api_key",
  "cached",
  "variant_name",
  "function_name",
  "episode_id",
  "search_query",
  "filters",
  "before",
  "after",
  "tags",
] as const;

export default function InferenceSearchBar({
  apiKeys = [],
}: {
  apiKeys?: KeyInfo[];
}) {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const config = useConfig();
  const functions = useAllFunctionConfigs();
  const query = parseInferenceQuery(searchParams);
  const variantName = searchParams.get("variant_name") ?? "";
  const functionName = searchParams.get("function_name");
  const [requestId, setRequestId] = useState(query.requestId);
  const [tags, setTags] = useState(query.tags);

  useEffect(() => {
    setRequestId(query.requestId);
  }, [query.requestId]);

  useEffect(() => {
    setTags(query.tags);
  }, [query.tags]);

  const providers = useMemo(() => providersFromConfig(config), [config]);
  const aliases = useMemo(
    () => config.model_aliases.map((alias) => alias.name),
    [config.model_aliases],
  );
  const modelOptions = useMemo(() => {
    const options = modelOptionsForProvider(providers, query.provider, aliases);
    if (variantName && !options.includes(variantName)) {
      return [variantName, ...options].sort();
    }
    return options;
  }, [providers, query.provider, aliases, variantName]);
  const apiKeyOptions = useMemo(
    () => apiKeysForSelect(apiKeys, query.apiKey),
    [apiKeys, query.apiKey],
  );

  const apply = (
    nextQuery: InferenceQueryValues,
    nextVariantName: string,
    nextFunctionName: string | null,
  ) => {
    const params = new URLSearchParams(searchParams);
    params.delete("before");
    params.delete("after");
    setOrDelete(params, "request_id", nextQuery.requestId.trim());
    setOrDelete(params, "provider", nextQuery.provider.trim());
    setOrDelete(params, "api_key", nextQuery.apiKey.trim());
    setOrDelete(params, "variant_name", nextVariantName.trim());
    setOrDelete(
      params,
      "start_time",
      fromDatetimeLocalValue(nextQuery.startTime) ?? "",
    );
    setOrDelete(
      params,
      "end_time",
      fromDatetimeLocalValue(nextQuery.endTime) ?? "",
    );
    if (nextQuery.cached !== "all") {
      params.set("cached", nextQuery.cached);
    } else {
      params.delete("cached");
    }
    setOrDelete(params, "function_name", nextFunctionName ?? "");
    setOrDelete(params, "tags", nextQuery.tags.trim());
    navigate(`?${params.toString()}`, { preventScrollReset: true });
  };

  useEffect(() => {
    if (requestId === query.requestId) return;
    const handle = window.setTimeout(() => {
      apply({ ...query, requestId, tags }, variantName, functionName);
    }, 400);
    return () => window.clearTimeout(handle);
    // apply is stable enough via URL-derived query/variant/function.
    // oxlint-disable-next-line react-hooks/exhaustive-deps
  }, [requestId]);

  useEffect(() => {
    if (tags === query.tags) return;
    const handle = window.setTimeout(() => {
      apply({ ...query, requestId, tags }, variantName, functionName);
    }, 400);
    return () => window.clearTimeout(handle);
    // oxlint-disable-next-line react-hooks/exhaustive-deps
  }, [tags]);

  const hasActive =
    JSON.stringify({ ...query, requestId, tags }) !==
      JSON.stringify(DEFAULT_INFERENCE_QUERY) ||
    variantName.length > 0 ||
    Boolean(functionName);

  return (
    <div className="mb-4 flex flex-wrap items-end gap-2 rounded-md border bg-muted/30 p-3">
      <Field label="Provider">
        <Select
          value={query.provider || "all"}
          onValueChange={(value) => {
            const provider = value === "all" ? "" : value;
            const nextModels = modelOptionsForProvider(
              providers,
              provider,
              aliases,
            );
            const nextVariant = nextModels.includes(variantName)
              ? variantName
              : "";
            apply(
              { ...query, requestId, tags, provider },
              nextVariant,
              functionName,
            );
          }}
        >
          <SelectTrigger className="h-8 w-[140px]" aria-label="Provider">
            <SelectValue placeholder="All Providers" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Providers</SelectItem>
            {providers.map((provider) => (
              <SelectItem key={provider.id} value={provider.id}>
                {provider.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </Field>
      <Field label="API key">
        <Select
          value={query.apiKey || "all"}
          onValueChange={(value) =>
            apply(
              {
                ...query,
                requestId,
                tags,
                apiKey: value === "all" ? "" : value,
              },
              variantName,
              functionName,
            )
          }
        >
          <SelectTrigger className="h-8 w-[220px]" aria-label="API key">
            <SelectValue placeholder="All API keys" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All API keys</SelectItem>
            {apiKeyOptions.map((key) => (
              <SelectItem key={key.public_id} value={key.public_id}>
                {formatApiKeyOption(key)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </Field>
      <Field label="Model">
        <Select
          value={variantName || "all"}
          onValueChange={(value) =>
            apply(
              { ...query, requestId, tags },
              value === "all" ? "" : value,
              functionName,
            )
          }
          disabled={modelOptions.length === 0 && !variantName}
        >
          <SelectTrigger className="h-8 w-[200px]" aria-label="Model">
            <SelectValue placeholder="All Models" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Models</SelectItem>
            {modelOptions.map((model) => (
              <SelectItem key={model} value={model}>
                {model}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </Field>
      <Field label="Function">
        <div className="flex items-center gap-1">
          <div className="w-[200px]">
            <FunctionSelector
              selected={functionName}
              onSelect={(name) =>
                apply({ ...query, requestId, tags }, variantName, name)
              }
              functions={functions}
              ariaLabel="Function"
            />
          </div>
          {functionName && (
            <Button
              type="button"
              variant="ghost"
              size="iconSm"
              className="h-8 w-8"
              aria-label="Clear function filter"
              onClick={() =>
                apply({ ...query, requestId, tags }, variantName, null)
              }
            >
              <X className="h-3.5 w-3.5" />
            </Button>
          )}
        </div>
      </Field>
      <Field label="Cache">
        <Select
          value={query.cached}
          onValueChange={(value) =>
            apply(
              {
                ...query,
                requestId,
                tags,
                cached: value as InferenceQueryValues["cached"],
              },
              variantName,
              functionName,
            )
          }
        >
          <SelectTrigger className="h-8 w-[120px]" aria-label="Cache">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All</SelectItem>
            <SelectItem value="true">Cached</SelectItem>
            <SelectItem value="false">Not cached</SelectItem>
          </SelectContent>
        </Select>
      </Field>
      <Field label="Time range">
        <div className="flex items-center gap-1">
          <Input
            type="datetime-local"
            className="h-8 w-[180px]"
            value={toDatetimeLocalValue(query.startTime)}
            onChange={(event) =>
              apply(
                { ...query, requestId, tags, startTime: event.target.value },
                variantName,
                functionName,
              )
            }
          />
          <span className="text-xs text-muted-foreground">–</span>
          <Input
            type="datetime-local"
            className="h-8 w-[180px]"
            value={toDatetimeLocalValue(query.endTime)}
            onChange={(event) =>
              apply(
                { ...query, requestId, tags, endTime: event.target.value },
                variantName,
                functionName,
              )
            }
          />
        </div>
      </Field>
      <Field label="ID">
        <Input
          className="h-8 w-[280px] font-mono text-xs"
          value={requestId}
          onChange={(event) => setRequestId(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              apply({ ...query, requestId, tags }, variantName, functionName);
            }
          }}
          onBlur={() => {
            if (requestId !== query.requestId) {
              apply({ ...query, requestId, tags }, variantName, functionName);
            }
          }}
          placeholder="inference, episode, or request id"
          aria-label="ID"
        />
      </Field>
      <Field label="Tags">
        <Input
          className="h-8 w-[220px] font-mono text-xs"
          value={tags}
          onChange={(event) => setTags(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              apply({ ...query, requestId, tags }, variantName, functionName);
            }
          }}
          onBlur={() => {
            if (tags !== query.tags) {
              apply({ ...query, requestId, tags }, variantName, functionName);
            }
          }}
          placeholder="env=prod,team=ml"
          aria-label="Tags"
        />
      </Field>
      {hasActive && (
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="h-8"
          onClick={() => {
            const params = new URLSearchParams(searchParams);
            for (const key of FILTER_PARAM_KEYS) {
              params.delete(key);
            }
            navigate(`?${params.toString()}`, { preventScrollReset: true });
          }}
        >
          <X className="mr-1 h-3.5 w-3.5" />
          Clear
        </Button>
      )}
    </div>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-[11px] text-muted-foreground">{label}</label>
      {children}
    </div>
  );
}

function setOrDelete(params: URLSearchParams, key: string, value: string) {
  if (value) {
    params.set(key, value);
  } else {
    params.delete(key);
  }
}

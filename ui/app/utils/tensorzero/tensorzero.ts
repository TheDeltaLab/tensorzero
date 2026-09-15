// Modified by Delta-AI under Apache 2.0
/*
TensorZero Client (for internal use only for now)

TODO(shuyangli): Figure out a way to generate the HTTP client, possibly from Schema.
*/

import { z } from "zod";
import { BaseTensorZeroClient } from "./base-client";
import { ZodJsonValueSchema } from "~/utils/clickhouse/common";
import type {
  ApplyConfigTomlRequest,
  ApplyConfigTomlResponse,
  CountFeedbackByTargetIdResponse,
  CountModelsResponse,
  CumulativeFeedbackTimeSeriesPoint,
  DemonstrationFeedbackRow,
  FeedbackRow,
  FunctionInferenceCount,
  GetConfigTomlResponse,
  GetDemonstrationFeedbackResponse,
  GetEpisodeInferenceCountResponse,
  GetFeedbackBoundsResponse,
  GetFeedbackByTargetIdResponse,
  GetFunctionThroughputByVariantResponse,
  GetInferencesRequest,
  GetInferencesResponse,
  GetInferencesProtectionRequest,
  GetModelInferencesResponse,
  GetModelLatencyResponse,
  GetModelUsageResponse,
  GetVariantUsageResponse,
  InferenceCountByVariant,
  InferenceCountResponse,
  InferenceProtectionResponse,
  InferenceRetentionConfig,
  InferenceStorageStatsResponse,
  InferenceWithFeedbackCountResponse,
  InferencesProtectionResponse,
  LatestFeedbackIdByMetricResponse,
  ListEpisodesResponse,
  ListFunctionsWithInferenceCountResponse,
  ListInferenceMetadataResponse,
  ListInferencesRequest,
  MetricsWithFeedbackResponse,
  StatusResponse,
  TableBoundsWithCount,
  TimeWindow,
  UiConfig,
  UpdateInferenceRetentionRequest,
  ValidateConfigTomlRequest,
  ValidateConfigTomlResponse,
  VariantPerformancesResponse,
  ClientInferenceParams,
  InferenceResponse,
  ResolveUuidResponse,
  AsyncTaskStatus,
  AsyncTaskStatusResponse,
  ListAsyncTasksResponse,
} from "~/types/tensorzero";
import type { AnalysisResponse } from "~/routes/observability/analysis/analysisQuery";

export interface SynapseAnalyticsRow {
  tag?: string | null;
  model_name: string;
  requests: number;
  input_tokens: number;
  output_tokens: number;
  avg_latency_ms: number | null;
  avg_ttft_ms: number | null;
  output_tps_excluding_ttft: number | null;
  kind: string;
}

export interface SynapseBalances {
  deepseek: unknown | null;
  openrouter: unknown | null;
}

export interface InferenceApiKeyOption {
  public_id: string;
  description?: string | null;
  disabled: boolean;
}

export interface DashboardSessionResponse {
  enabled: boolean;
  allowed: boolean;
  email?: string | null;
  is_admin: boolean;
}

export interface DashboardUser {
  email: string;
  is_admin: boolean;
  created_at: string;
  updated_at: string;
  created_by?: string | null;
}

function dashboardEmailHeaders(
  email: string | null,
): Record<string, string> | undefined {
  if (!email) return undefined;
  return { "X-Auth-Request-Email": email };
}

/**
 * Feedback requests attach a metric value to a given inference or episode.
 */
export const FeedbackRequestSchema = z.object({
  dryrun: z.boolean().optional(),
  episode_id: z.string().nullable(),
  inference_id: z.string().nullable(),
  metric_name: z.string(),
  tags: z.record(z.string()).optional(),
  value: ZodJsonValueSchema,
  internal: z.boolean().optional(),
});
export type FeedbackRequest = z.infer<typeof FeedbackRequestSchema>;

export const FeedbackResponseSchema = z.object({
  feedback_id: z.string(),
});
export type FeedbackResponse = z.infer<typeof FeedbackResponseSchema>;

/**
 * Response type for getCumulativeFeedbackTimeseries endpoint
 */
export interface GetCumulativeFeedbackTimeseriesResponse {
  timeseries: CumulativeFeedbackTimeSeriesPoint[];
}

/**
 * A client for calling the TensorZero Gateway inference and feedback endpoints.
 */
export class TensorZeroClient extends BaseTensorZeroClient {
  /**
   * Performs an inference request.
   * @param request - The inference request payload.
   * @returns A promise that resolves with the inference response.
   * @throws Error if streaming is requested (not supported) or if the request fails.
   */
  async inference(request: ClientInferenceParams): Promise<InferenceResponse> {
    if (request.stream) {
      // TODO(#5394): support streaming inference.
      throw new Error("Streaming inference is not supported from the UI");
    }
    const response = await this.fetch("/inference", {
      method: "POST",
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    const body = (await response.json()) as InferenceResponse;
    return body;
  }

  async embeddings(request: {
    model: string;
    input: string | string[];
  }): Promise<unknown> {
    const response = await this.fetch("/v1/embeddings", {
      method: "POST",
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return response.json();
  }

  async rerank(request: {
    model: string;
    query: string;
    documents: string[];
    top_n?: number;
  }): Promise<unknown> {
    const response = await this.fetch("/v1/rerank", {
      method: "POST",
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return response.json();
  }

  async chatCompletions(request: {
    model: string;
    messages: Array<{ role: string; content: string }>;
    temperature?: number;
    max_tokens?: number;
  }): Promise<unknown> {
    const response = await this.fetch("/v1/chat/completions", {
      method: "POST",
      body: JSON.stringify({
        ...request,
        stream: false,
      }),
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return response.json();
  }

  async getSynapseAnalytics(
    from: string,
    to: string,
    options: { tags?: string; groupByTag?: string } = {},
  ): Promise<{ data: SynapseAnalyticsRow[] }> {
    const params = new URLSearchParams({ from, to });
    if (options.tags?.trim()) {
      params.set("tags", options.tags.trim());
    }
    if (options.groupByTag?.trim()) {
      params.set("group_by_tag", options.groupByTag.trim());
    }
    const response = await this.fetch(
      `/internal/synapse/analytics?${params.toString()}`,
      { method: "GET" },
    );
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as { data: SynapseAnalyticsRow[] };
  }

  async getSynapseAnalysis(options: {
    range: string;
    /// Absolute RFC 3339 window; takes precedence over `range` on the server.
    from?: string;
    to?: string;
    kind: string;
    apiKey?: string;
    model?: string;
    cacheMissOnly?: boolean;
    tagKey?: string;
  }): Promise<AnalysisResponse> {
    const params = new URLSearchParams({
      range: options.range,
      kind: options.kind,
    });
    if (options.from) {
      params.set("from", options.from);
    }
    if (options.to) {
      params.set("to", options.to);
    }
    if (options.apiKey?.trim()) {
      params.set("api_key", options.apiKey.trim());
    }
    if (options.model?.trim()) {
      params.set("model", options.model.trim());
    }
    if (options.cacheMissOnly) {
      params.set("cache_miss_only", "true");
    }
    if (options.tagKey?.trim()) {
      params.set("tag_key", options.tagKey.trim());
    }
    const response = await this.fetch(
      `/internal/synapse/analysis?${params.toString()}`,
      { method: "GET" },
    );
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as AnalysisResponse;
  }

  /**
   * API keys for the Inferences filter dropdown: native keys, imported Synapse
   * keys, and public ids already stored on inference tags.
   */
  async listInferenceApiKeys(): Promise<InferenceApiKeyOption[]> {
    const response = await this.fetch("/internal/inference_api_keys", {
      method: "GET",
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    const body = (await response.json()) as {
      api_keys: InferenceApiKeyOption[];
    };
    return body.api_keys ?? [];
  }

  /**
   * Async inference tasks from the gateway's durable queue, for the async
   * tasks dashboard page. (Delta-AI fork: restored after the #60 strip.)
   */
  async listAsyncTasks(options: {
    limit: number;
    offset: number;
    status?: AsyncTaskStatus;
  }): Promise<ListAsyncTasksResponse> {
    const params = new URLSearchParams({
      limit: options.limit.toString(),
      offset: options.offset.toString(),
    });
    if (options.status) {
      params.set("status", options.status);
    }
    const response = await this.fetch(
      `/internal/async_tasks?${params.toString()}`,
      { method: "GET" },
    );
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as ListAsyncTasksResponse;
  }

  /**
   * Detail for one async inference task (public API shape: status plus the
   * final response or error payload).
   */
  async getAsyncTask(taskId: string): Promise<AsyncTaskStatusResponse> {
    const response = await this.fetch(`/v1/async_tasks/${taskId}`, {
      method: "GET",
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as AsyncTaskStatusResponse;
  }

  async getDashboardSession(
    email: string | null,
  ): Promise<DashboardSessionResponse> {
    const response = await this.fetch("/internal/dashboard/session", {
      method: "GET",
      headers: dashboardEmailHeaders(email),
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as DashboardSessionResponse;
  }

  async listDashboardUsers(email: string): Promise<DashboardUser[]> {
    const response = await this.fetch("/internal/dashboard/users", {
      method: "GET",
      headers: dashboardEmailHeaders(email),
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    const body = (await response.json()) as { users: DashboardUser[] };
    return body.users ?? [];
  }

  async createDashboardUser(
    actorEmail: string,
    payload: { email: string; is_admin: boolean },
  ): Promise<DashboardUser> {
    const response = await this.fetch("/internal/dashboard/users", {
      method: "POST",
      headers: dashboardEmailHeaders(actorEmail),
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as DashboardUser;
  }

  async updateDashboardUser(
    actorEmail: string,
    payload: { email: string; is_admin: boolean },
  ): Promise<DashboardUser> {
    const response = await this.fetch("/internal/dashboard/users", {
      method: "PATCH",
      headers: dashboardEmailHeaders(actorEmail),
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as DashboardUser;
  }

  async deleteDashboardUser(actorEmail: string, email: string): Promise<void> {
    const response = await this.fetch("/internal/dashboard/users/delete", {
      method: "POST",
      headers: dashboardEmailHeaders(actorEmail),
      body: JSON.stringify({ email }),
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
  }

  async getSynapseUsageExport(from: string, to: string): Promise<string> {
    const params = new URLSearchParams({ from, to });
    const response = await this.fetch(
      `/internal/synapse/usage_export?${params.toString()}`,
      { method: "GET" },
    );
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return response.text();
  }

  async getSynapseBalances(): Promise<SynapseBalances> {
    const response = await this.fetch("/internal/synapse/balances", {
      method: "GET",
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as SynapseBalances;
  }

  /**
   * Sends feedback for a particular inference or episode.
   * @param request - The feedback request payload.
   * @returns A promise that resolves with the feedback response.
   */
  async feedback(request: FeedbackRequest): Promise<FeedbackResponse> {
    const response = await this.fetch("/feedback", {
      method: "POST",
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as FeedbackResponse;
  }

  /**
   * Queries feedback for a given target ID with pagination support.
   * @param targetId - The target ID (inference_id or episode_id) to query feedback for
   * @param options - Optional pagination parameters
   * @returns A promise that resolves with a list of feedback rows
   * @throws Error if the request fails
   */
  async getFeedbackByTargetId(
    targetId: string,
    options?: {
      before?: string;
      after?: string;
      limit?: number;
    },
  ): Promise<FeedbackRow[]> {
    const params = new URLSearchParams();
    if (options?.before) params.set("before", options.before);
    if (options?.after) params.set("after", options.after);
    if (options?.limit) params.set("limit", options.limit.toString());

    const queryString = params.toString();
    const endpoint = `/internal/feedback/${encodeURIComponent(targetId)}${queryString ? `?${queryString}` : ""}`;
    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    const body = (await response.json()) as GetFeedbackByTargetIdResponse;
    return body.feedback;
  }

  /**
   * Gets demonstration feedback for a given inference.
   * @param inferenceId - The inference ID to get demonstration feedback for
   * @param options - Optional pagination parameters
   * @returns A promise that resolves with a list of demonstration feedback rows
   * @throws Error if the request fails
   */
  async getDemonstrationFeedback(
    inferenceId: string,
    options?: {
      before?: string;
      after?: string;
      limit?: number;
    },
  ): Promise<DemonstrationFeedbackRow[]> {
    const params = new URLSearchParams();
    if (options?.before) params.set("before", options.before);
    if (options?.after) params.set("after", options.after);
    if (options?.limit) params.set("limit", options.limit.toString());

    const queryString = params.toString();
    const endpoint = `/internal/feedback/${encodeURIComponent(inferenceId)}/demonstrations${queryString ? `?${queryString}` : ""}`;
    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    const body = (await response.json()) as GetDemonstrationFeedbackResponse;
    return body.feedback;
  }

  /**
   * Gets model usage timeseries data.
   * @param timeWindow The time window granularity for grouping data
   * @param maxPeriods Maximum number of time periods to return
   * @returns A promise that resolves with the model usage timeseries data
   * @throws Error if the request fails
   */
  async getModelUsageTimeseries(
    timeWindow: TimeWindow,
    maxPeriods: number,
  ): Promise<GetModelUsageResponse> {
    const params = new URLSearchParams({
      time_window: timeWindow,
      max_periods: maxPeriods.toString(),
    });
    const response = await this.fetch(
      `/internal/models/usage?${params.toString()}`,
      {
        method: "GET",
      },
    );
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as GetModelUsageResponse;
  }

  /**
   * Gets model latency quantile distributions.
   * @param timeWindow The time window for aggregating latency data
   * @returns A promise that resolves with the model latency quantiles
   * @throws Error if the request fails
   */
  async getModelLatencyQuantiles(
    timeWindow: TimeWindow,
  ): Promise<GetModelLatencyResponse> {
    const params = new URLSearchParams({
      time_window: timeWindow,
    });
    const response = await this.fetch(
      `/internal/models/latency?${params.toString()}`,
      {
        method: "GET",
      },
    );
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as GetModelLatencyResponse;
  }

  async getObject(path: string): Promise<string> {
    const endpoint = `/internal/object_storage?path=${encodeURIComponent(path)}`;
    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return response.text();
  }

  async status(): Promise<StatusResponse> {
    const response = await this.fetch("/status", { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as StatusResponse;
  }

  /**
   * Lists inferences with optional filtering, pagination, and sorting.
   * Uses the public v1 API endpoint.
   * @param request - The list inferences request parameters
   * @returns A promise that resolves with the inferences response
   * @throws Error if the request fails
   */
  async listInferences(
    request: ListInferencesRequest,
  ): Promise<GetInferencesResponse> {
    const response = await this.fetch("/v1/inferences/list_inferences", {
      method: "POST",
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as GetInferencesResponse;
  }

  /**
   * Retrieves specific inferences by their IDs.
   * Uses the public v1 API endpoint.
   * @param request - The get inferences request containing IDs and optional filters
   * @returns A promise that resolves with the inferences response
   * @throws Error if the request fails
   */
  async getInferences(
    request: GetInferencesRequest,
  ): Promise<GetInferencesResponse> {
    const response = await this.fetch("/v1/inferences/get_inferences", {
      method: "POST",
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as GetInferencesResponse;
  }

  /**
   * Fetches the gateway configuration for the UI.
   * @returns A promise that resolves with the UiConfig object
   * @throws Error if the request fails
   */
  async getUiConfig(): Promise<UiConfig> {
    const response = await this.fetch("/internal/ui_config", { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as UiConfig;
  }

  async getUiConfigByHash(hash: string): Promise<UiConfig> {
    const response = await this.fetch(
      `/internal/ui_config/${encodeURIComponent(hash)}`,
      { method: "GET" },
    );
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as UiConfig;
  }

  async getConfigToml(): Promise<GetConfigTomlResponse> {
    const response = await this.fetch("/internal/config_toml", {
      method: "GET",
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as GetConfigTomlResponse;
  }

  async applyConfigToml(
    request: ApplyConfigTomlRequest,
  ): Promise<ApplyConfigTomlResponse> {
    const response = await this.fetch("/internal/config_toml/apply", {
      method: "POST",
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as ApplyConfigTomlResponse;
  }

  async validateConfigToml(
    request: ValidateConfigTomlRequest,
  ): Promise<ValidateConfigTomlResponse> {
    const response = await this.fetch("/internal/config_toml/validate", {
      method: "POST",
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as ValidateConfigTomlResponse;
  }

  /**
   * Fetches inference count for a function, optionally filtered by variant or grouped by variant.
   * @param functionName - The name of the function to get count for
   * @param options - Optional parameters for filtering or grouping
   * @param options.variantName - Optional variant name to filter by
   * @param options.groupBy - Optional grouping (e.g., "variant" to get counts per variant)
   * @returns A promise that resolves with the inference count
   * @throws Error if the request fails
   */
  async getInferenceCount(
    functionName: string,
    options?: { variantName?: string; groupBy?: "variant" },
  ): Promise<InferenceCountResponse> {
    const searchParams = new URLSearchParams();
    if (options?.variantName) {
      searchParams.append("variant_name", options.variantName);
    }
    if (options?.groupBy) {
      searchParams.append("group_by", options.groupBy);
    }
    const queryString = searchParams.toString();
    const endpoint = `/internal/functions/${encodeURIComponent(functionName)}/inference_count${queryString ? `?${queryString}` : ""}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as InferenceCountResponse;
  }

  /**
   * Fetches the variants used for a function.
   * @param functionName - The name of the function to get variants for
   * @returns A promise that resolves with the variants used for the function
   * @throws Error if the request fails
   */
  async getUsedVariants(functionName: string): Promise<string[]> {
    const response = await this.getInferenceCount(functionName, {
      groupBy: "variant",
    });

    return (response.count_by_variant ?? []).map(
      (v: InferenceCountByVariant) => v.variant_name,
    );
  }

  /**
   * Lists all functions with their inference counts, ordered by most recent inference.
   * @returns A promise that resolves with the function inference counts
   * @throws Error if the request fails
   */
  async listFunctionsWithInferenceCount(): Promise<FunctionInferenceCount[]> {
    const endpoint = `/internal/functions/inference_counts`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    const body =
      (await response.json()) as ListFunctionsWithInferenceCountResponse;
    return body.functions;
  }

  /**
   * Fetches feedback counts for a function and metric.
   * @param functionName - The name of the function to get count for
   * @param metricName - The name of the metric to get count for (or "demonstration")
   * @param threshold - Optional threshold for float metrics (defaults to 0)
   * @returns A promise that resolves with the feedback and curated inference counts
   * @throws Error if the request fails
   */
  async getFeedbackCount(
    functionName: string,
    metricName: string,
    threshold?: number,
  ): Promise<InferenceWithFeedbackCountResponse> {
    const searchParams = new URLSearchParams();
    if (threshold !== undefined) {
      searchParams.append("threshold", threshold.toString());
    }
    const queryString = searchParams.toString();
    const endpoint = `/internal/functions/${encodeURIComponent(functionName)}/inference_count/${encodeURIComponent(metricName)}${queryString ? `?${queryString}` : ""}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as InferenceWithFeedbackCountResponse;
  }

  /**
   * Fetches function throughput data grouped by variant and time period.
   * @param functionName - The name of the function to get throughput data for
   * @param timeWindow - The time granularity for grouping data (minute, hour, day, week, month, cumulative)
   * @param maxPeriods - Maximum number of time periods to return
   * @returns A promise that resolves with the throughput data
   * @throws Error if the request fails
   */
  async getFunctionThroughputByVariant(
    functionName: string,
    timeWindow: TimeWindow,
    maxPeriods: number,
  ): Promise<GetFunctionThroughputByVariantResponse> {
    const searchParams = new URLSearchParams({
      time_window: timeWindow,
      max_periods: maxPeriods.toString(),
    });
    const endpoint = `/internal/functions/${encodeURIComponent(functionName)}/throughput_by_variant?${searchParams.toString()}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as GetFunctionThroughputByVariantResponse;
  }

  /**
   * Gets variant usage timeseries data for a function.
   * @param functionName - The function to get variant usage for
   * @param timeWindow - The time window granularity
   * @param maxPeriods - Maximum number of periods to return
   * @returns A promise that resolves with variant usage data
   * @throws Error if the request fails
   */
  async getVariantUsageTimeseries(
    functionName: string,
    timeWindow: TimeWindow,
    maxPeriods: number,
  ): Promise<GetVariantUsageResponse> {
    const searchParams = new URLSearchParams({
      time_window: timeWindow,
      max_periods: maxPeriods.toString(),
    });
    const endpoint = `/internal/functions/${encodeURIComponent(functionName)}/variant_usage?${searchParams.toString()}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as GetVariantUsageResponse;
  }

  /**
   * Fetches metrics with feedback for a function, optionally filtered by variant.
   * @param functionName - The name of the function to get metrics for
   * @param variantName - Optional variant name to filter by
   * @returns A promise that resolves with metrics and their feedback counts
   * @throws Error if the request fails
   */
  async getFunctionMetricsWithFeedback(
    functionName: string,
    variantName?: string,
  ): Promise<MetricsWithFeedbackResponse> {
    const searchParams = new URLSearchParams();
    if (variantName) {
      searchParams.append("variant_name", variantName);
    }
    const queryString = searchParams.toString();
    const endpoint = `/internal/functions/${encodeURIComponent(functionName)}/metrics${queryString ? `?${queryString}` : ""}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as MetricsWithFeedbackResponse;
  }

  /**
   * Fetches variant performance statistics for a function and metric.
   * @param functionName - The name of the function to get performance stats for
   * @param metricName - The name of the metric to compute performance for
   * @param timeWindow - Time granularity for grouping performance data
   * @param variantName - Optional variant name to filter by
   * @returns A promise that resolves with variant performance statistics
   * @throws Error if the request fails
   */
  async getVariantPerformances(
    functionName: string,
    metricName: string,
    timeWindow: TimeWindow,
    variantName?: string,
  ): Promise<VariantPerformancesResponse> {
    const searchParams = new URLSearchParams();
    searchParams.append("metric_name", metricName);
    searchParams.append("time_window", timeWindow);
    if (variantName) {
      searchParams.append("variant_name", variantName);
    }
    const queryString = searchParams.toString();
    const endpoint = `/internal/functions/${encodeURIComponent(functionName)}/variant_performances?${queryString}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as VariantPerformancesResponse;
  }

  /**
   * Fetches model inferences for a given inference ID.
   * @param inferenceId - The UUID of the inference to get model inferences for
   * @returns A promise that resolves with the model inferences response
   * @throws Error if the request fails
   */
  async getModelInferences(
    inferenceId: string,
  ): Promise<GetModelInferencesResponse> {
    const endpoint = `/internal/model_inferences/${encodeURIComponent(inferenceId)}`;
    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as GetModelInferencesResponse;
  }

  /**
   * Counts the number of distinct models used.
   * @returns A promise that resolves with the count of distinct models
   * @throws Error if the request fails
   */
  async countDistinctModelsUsed(): Promise<CountModelsResponse> {
    const response = await this.fetch("/internal/models/count", {
      method: "GET",
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as CountModelsResponse;
  }

  /**
   * Lists inference metadata with optional cursor-based pagination and filtering.
   * @param params - Optional pagination and filter parameters
   * @param params.before - Cursor to fetch records before this ID (mutually exclusive with after)
   * @param params.after - Cursor to fetch records after this ID (mutually exclusive with before)
   * @param params.limit - Maximum number of records to return
   * @param params.function_name - Optional function name to filter by
   * @param params.variant_name - Optional variant name to filter by
   * @param params.episode_id - Optional episode ID to filter by
   * @returns A promise that resolves with the inference metadata response
   * @throws Error if the request fails
   */
  async listInferenceMetadata(params?: {
    before?: string;
    after?: string;
    limit?: number;
    function_name?: string | null;
    variant_name?: string | null;
    episode_id?: string | null;
  }): Promise<ListInferenceMetadataResponse> {
    const searchParams = new URLSearchParams();
    if (params?.before) {
      searchParams.append("before", params.before);
    }
    if (params?.after) {
      searchParams.append("after", params.after);
    }
    if (params?.limit !== undefined) {
      searchParams.append("limit", params.limit.toString());
    }
    if (params?.function_name) {
      searchParams.append("function_name", params.function_name);
    }
    if (params?.variant_name) {
      searchParams.append("variant_name", params.variant_name);
    }
    if (params?.episode_id) {
      searchParams.append("episode_id", params.episode_id);
    }
    const queryString = searchParams.toString();
    const endpoint = `/internal/inference_metadata${queryString ? `?${queryString}` : ""}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as ListInferenceMetadataResponse;
  }

  /**
   * Lists episodes with pagination support.
   * @param limit - Maximum number of episodes to return
   * @param before - Return episodes before this episode_id (for pagination)
   * @param after - Return episodes after this episode_id (for pagination)
   * @returns A promise that resolves with an array of episodes
   * @throws Error if the request fails
   */
  async listEpisodes(
    limit: number,
    before?: string,
    after?: string,
  ): Promise<ListEpisodesResponse> {
    const searchParams = new URLSearchParams();
    searchParams.append("limit", limit.toString());
    if (before) {
      searchParams.append("before", before);
    }
    if (after) {
      searchParams.append("after", after);
    }
    const queryString = searchParams.toString();
    const endpoint = `/internal/episodes?${queryString}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as ListEpisodesResponse;
  }

  /**
   * Queries episode table bounds (first_id, last_id, and count).
   * @returns A promise that resolves with the bounds information
   * @throws Error if the request fails
   */
  async queryEpisodeTableBounds(): Promise<TableBoundsWithCount> {
    const endpoint = `/internal/episodes/bounds`;
    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as TableBoundsWithCount;
  }

  /**
   * Gets inference counts for a specific episode.
   * @param episode_id - The UUID of the episode
   * @returns A promise that resolves with the inference counts
   * @throws Error if the request fails
   */
  async getEpisodeInferenceCount(
    episode_id: string,
  ): Promise<GetEpisodeInferenceCountResponse> {
    const endpoint = `/internal/episodes/${episode_id}/inference_count`;
    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as GetEpisodeInferenceCountResponse;
  }

  /**
   * Queries feedback bounds for a given target ID.
   * @param targetId - The target ID (inference_id or episode_id) to query feedback bounds for
   * @returns A promise that resolves with the feedback bounds across all feedback types
   * @throws Error if the request fails
   */
  async getFeedbackBoundsByTargetId(
    targetId: string,
  ): Promise<GetFeedbackBoundsResponse> {
    const endpoint = `/internal/feedback/${encodeURIComponent(targetId)}/bounds`;
    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as GetFeedbackBoundsResponse;
  }

  /**
   * Queries the latest feedback ID for each metric for a given target.
   * @param targetId - The target ID (inference_id or episode_id) to query feedback for
   * @returns A promise that resolves with a mapping of metric names to their latest feedback IDs
   * @throws Error if the request fails
   */
  async getLatestFeedbackIdByMetric(
    targetId: string,
  ): Promise<Record<string, string>> {
    const endpoint = `/internal/feedback/${encodeURIComponent(targetId)}/latest_id_by_metric`;
    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    const body = (await response.json()) as LatestFeedbackIdByMetricResponse;
    // Convert optional values to non-optional (ts-rs generates HashMap as optional, but values are always present)
    return Object.fromEntries(
      Object.entries(body.feedback_id_by_metric).filter(
        (entry): entry is [string, string] => entry[1] !== undefined,
      ),
    );
  }

  /**
   * Queries the count of feedback for a given target ID.
   * @param targetId - The target ID (inference_id or episode_id) to count feedback for
   * @returns A promise that resolves with the feedback count
   * @throws Error if the request fails
   */
  async countFeedbackByTargetId(targetId: string): Promise<number> {
    const endpoint = `/internal/feedback/${encodeURIComponent(targetId)}/count`;
    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    const body = (await response.json()) as CountFeedbackByTargetIdResponse;
    return Number(body.count);
  }

  /**
   * Gets cumulative feedback time series for a function and metric.
   * @param functionName - The name of the function to get feedback for
   * @param metricName - The name of the metric to get feedback for
   * @param timeWindow - The time window granularity for grouping data
   * @param maxPeriods - Maximum number of time periods to return
   * @param variantNames - Optional array of variant names to filter by
   * @returns A promise that resolves with cumulative feedback time series data
   * @throws Error if the request fails
   */
  async getCumulativeFeedbackTimeseries(params: {
    function_name: string;
    metric_name: string;
    time_window: TimeWindow;
    max_periods: number;
    variant_names?: string[];
  }): Promise<CumulativeFeedbackTimeSeriesPoint[]> {
    const searchParams = new URLSearchParams({
      function_name: params.function_name,
      metric_name: params.metric_name,
      time_window: params.time_window,
      max_periods: params.max_periods.toString(),
    });
    if (params.variant_names && params.variant_names.length > 0) {
      searchParams.append("variant_names", params.variant_names.join(","));
    }
    const endpoint = `/internal/feedback/timeseries?${searchParams.toString()}`;
    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    const body =
      (await response.json()) as GetCumulativeFeedbackTimeseriesResponse;
    return body.timeseries;
  }

  /**
   * Gets per-table storage statistics and the inference retention configuration.
   * Requires Postgres.
   */
  async getInferenceStorageStats(): Promise<InferenceStorageStatsResponse> {
    const response = await this.fetch("/internal/inference_storage/stats", {
      method: "GET",
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as InferenceStorageStatsResponse;
  }

  /**
   * Updates the inference retention configuration. A number keeps data for
   * that many days; omitting a field keeps data forever (deletes the key).
   */
  async updateInferenceRetention(
    request: UpdateInferenceRetentionRequest,
  ): Promise<InferenceRetentionConfig> {
    const response = await this.fetch("/internal/inference_storage/retention", {
      method: "POST",
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as InferenceRetentionConfig;
  }

  /**
   * Protects (or unprotects) an inference from retention cleanup.
   * @param inferenceId - The inference UUID
   * @param protected_ - Whether the inference should be protected
   */
  async setInferenceProtection(
    inferenceId: string,
    protected_: boolean,
  ): Promise<InferenceProtectionResponse> {
    const response = await this.fetch(
      `/internal/inferences/${encodeURIComponent(inferenceId)}/protection`,
      {
        method: "POST",
        body: JSON.stringify({ protected: protected_ }),
      },
    );
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as InferenceProtectionResponse;
  }

  /**
   * Gets the protection state for a batch of inferences (max 1000 ids).
   * Only protected inferences are returned.
   */
  async getInferencesProtection(
    ids: string[],
  ): Promise<InferencesProtectionResponse> {
    const response = await this.fetch("/internal/inferences/protection", {
      method: "POST",
      body: JSON.stringify({ ids } satisfies GetInferencesProtectionRequest),
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as InferencesProtectionResponse;
  }

  /**
   * Resolves a UUID to determine what type(s) of object it represents.
   * @param id - The UUID to resolve
   * @returns A promise that resolves with the resolved object types
   * @throws Error if the request fails
   */
  async resolveUuid(id: string): Promise<ResolveUuidResponse> {
    const endpoint = `/internal/resolve_uuid/${encodeURIComponent(id)}`;
    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as ResolveUuidResponse;
  }
}

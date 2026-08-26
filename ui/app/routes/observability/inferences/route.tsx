// Modified by Delta-AI under Apache 2.0
import { listInferencesWithPagination } from "~/utils/clickhouse/inference.server";
import type { Route } from "./+types/route";
import InferencesTable, {
  type InferenceListRow,
  type InferencesData,
} from "./InferencesTable";
import { data } from "react-router";
import InferenceSearchBar from "./InferenceSearchBar";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import type { InferenceFilter, StoredInference } from "~/types/tensorzero";
import type { InferenceApiKeyOption } from "~/utils/tensorzero";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { logger } from "~/utils/logger";
import {
  buildDimensionalFilter,
  combineFilters,
  COST_TAG,
  INPUT_TOKENS_TAG,
  isUuid,
  mergeUsage,
  OUTPUT_TOKENS_TAG,
  parseInferenceQuery,
  requestIdTagFilter,
  toFiniteNumber,
  usageFromModelInferences,
  usageFromTags,
} from "./inferenceQuery";

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const before = url.searchParams.get("before");
  const after = url.searchParams.get("after");
  const limit = Number(url.searchParams.get("limit")) || 10;
  if (limit > 100) {
    throw data("Limit cannot exceed 100", { status: 400 });
  }

  const function_name = url.searchParams.get("function_name") || undefined;
  const variant_name = url.searchParams.get("variant_name") || undefined;
  const episode_id = url.searchParams.get("episode_id") || undefined;
  const search_query = url.searchParams.get("search_query") || undefined;

  const filtersParam = url.searchParams.get("filters");
  let advancedFilters: InferenceFilter | undefined;
  if (filtersParam) {
    try {
      advancedFilters = JSON.parse(filtersParam) as InferenceFilter;
    } catch {
      advancedFilters = undefined;
    }
  }

  const query = parseInferenceQuery(url.searchParams);
  const requestId = query.requestId.trim();
  const barFilters = buildDimensionalFilter(query, {
    includeRequestIdTags: false,
  });
  const sharedFilters = combineFilters(barFilters, advancedFilters);
  const apiKeysPromise = (async (): Promise<InferenceApiKeyOption[]> => {
    try {
      return await getTensorZeroClient().listInferenceApiKeys();
    } catch (error) {
      logger.error("Failed to list API keys for inference filters", error);
      return [];
    }
  })();

  const client = getTensorZeroClient();
  const totalInferencesPromise = client
    .listFunctionsWithInferenceCount()
    .then((countsInfo) =>
      countsInfo.reduce((acc, curr) => acc + curr.inference_count, 0),
    );

  const inferencesDataPromise: Promise<InferencesData> = (async () => {
    const listPage = async (extra: {
      episode_id?: string;
      filters?: InferenceFilter;
    }) =>
      listInferencesWithPagination({
        before: before || undefined,
        after: after || undefined,
        limit,
        function_name,
        variant_name,
        search_query,
        episode_id: extra.episode_id,
        filters: extra.filters,
      });

    const attach = async (
      inferences: InferenceListRow[],
      hasNextPage: boolean,
      hasPreviousPage: boolean,
    ): Promise<InferencesData> => ({
      inferences: await attachModelUsage(inferences),
      hasNextPage,
      hasPreviousPage,
    });

    if (isUuid(requestId)) {
      try {
        const byId = await client.getInferences({
          ids: [requestId],
          output_source: "inference",
        });
        const extra = byId.inferences[0];
        if (extra) {
          return attach([storedToRow(extra)], false, false);
        }
      } catch {
        // Not an inference id; try episode id, then request-id tags.
      }

      const byEpisode = await listPage({
        episode_id: requestId,
        filters: sharedFilters,
      });
      if (
        byEpisode.inferences.length > 0 ||
        byEpisode.hasNextPage ||
        byEpisode.hasPreviousPage
      ) {
        return attach(
          byEpisode.inferences.map(storedToRow),
          byEpisode.hasNextPage,
          byEpisode.hasPreviousPage,
        );
      }
    }

    const inferenceResult = await listPage({
      episode_id,
      filters: combineFilters(sharedFilters, requestIdTagFilter(requestId)),
    });

    return attach(
      inferenceResult.inferences.map(storedToRow),
      inferenceResult.hasNextPage,
      inferenceResult.hasPreviousPage,
    );
  })();

  return {
    inferencesData: inferencesDataPromise,
    totalInferences: totalInferencesPromise,
    limit,
    apiKeys: await apiKeysPromise,
  };
}

function storedToRow(inf: StoredInference): InferenceListRow {
  const usage = usageFromTags(inf.tags);
  return {
    id: inf.inference_id,
    episode_id: inf.episode_id,
    function_name: inf.function_name,
    variant_name: inf.variant_name,
    function_type: inf.type,
    snapshot_hash: inf.snapshot_hash,
    tags: inf.tags,
    processing_time_ms: toFiniteNumber(inf.processing_time_ms),
    ttft_ms: toFiniteNumber(inf.ttft_ms),
    input_tokens: usage.input_tokens ?? undefined,
    output_tokens: usage.output_tokens ?? undefined,
    cost: usage.cost,
    currency: usage.currency,
  };
}

function hasUsageTags(tags: Record<string, string> | undefined): boolean {
  if (!tags) return false;
  return Boolean(
    tags[INPUT_TOKENS_TAG] || tags[OUTPUT_TOKENS_TAG] || tags[COST_TAG],
  );
}

async function attachModelUsage(
  inferences: InferenceListRow[],
): Promise<InferenceListRow[]> {
  const missing = inferences.filter((row) => !hasUsageTags(row.tags));
  if (missing.length === 0) return inferences;

  const client = getTensorZeroClient();
  const extras = await Promise.all(
    missing.map(async (row) => {
      try {
        const response = await client.getModelInferences(row.id);
        return [
          row.id,
          usageFromModelInferences(response.model_inferences),
        ] as const;
      } catch {
        return [row.id, undefined] as const;
      }
    }),
  );
  const byId = new Map(extras);
  return inferences.map((row) => {
    const extra = byId.get(row.id);
    if (!extra) return row;
    const merged = mergeUsage(
      {
        input_tokens: row.input_tokens,
        output_tokens: row.output_tokens,
        cost: row.cost,
        currency: row.currency,
      },
      extra,
    );
    return {
      ...row,
      input_tokens: merged.input_tokens ?? undefined,
      output_tokens: merged.output_tokens ?? undefined,
      cost: merged.cost ?? undefined,
      currency: merged.currency ?? undefined,
    };
  });
}

export default function InferencesPage({ loaderData }: Route.ComponentProps) {
  const { inferencesData, totalInferences, limit, apiKeys } = loaderData;

  return (
    <PageLayout>
      <PageHeader heading="Inferences" count={totalInferences} />
      <SectionLayout>
        <InferenceSearchBar apiKeys={apiKeys} />
        <InferencesTable data={inferencesData} limit={limit} />
      </SectionLayout>
    </PageLayout>
  );
}

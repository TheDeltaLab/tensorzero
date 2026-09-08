// Modified by Delta-AI under Apache 2.0
import { data } from "react-router";
import type { StoredInference, InferenceFilter } from "~/types/tensorzero";
import { logger } from "~/utils/logger";
import { getTensorZeroClient } from "../tensorzero.server";
import { isTensorZeroServerError } from "../tensorzero";
import { applyPaginationLogic } from "../pagination";

/**
 * Result type for listInferencesWithPagination with pagination info.
 */
export type ListInferencesResult = {
  inferences: StoredInference[];
  hasNextPage: boolean;
  hasPreviousPage: boolean;
};

/**
 * Lists inferences using the public v1 API with cursor-based pagination.
 * Implements the limit+1 pattern to detect if there are more pages.
 *
 * @param params - Query parameters for listing inferences
 * @returns Inferences with pagination info (hasNextPage, hasPreviousPage)
 */
export async function listInferencesWithPagination(params: {
  limit: number;
  before?: string; // UUIDv7 string - get inferences before this ID (going to older)
  after?: string; // UUIDv7 string - get inferences after this ID (going to newer)
  offset?: number; // number of inferences to skip (page-number based pagination)
  function_name?: string;
  variant_name?: string;
  episode_id?: string;
  filters?: InferenceFilter;
  search_query?: string;
}): Promise<ListInferencesResult> {
  const {
    limit,
    before,
    after,
    offset,
    function_name,
    variant_name,
    episode_id,
    filters,
    search_query,
  } = params;

  if (before && after) {
    throw new Error("Cannot specify both 'before' and 'after' parameters");
  }
  if (offset !== undefined && (before || after)) {
    throw new Error("Cannot specify 'offset' together with 'before'/'after'");
  }

  try {
    const client = getTensorZeroClient();

    // Request limit + 1 to detect if there are more pages
    const response = await client.listInferences({
      output_source: "inference",
      limit: limit + 1,
      before,
      after,
      offset,
      function_name,
      variant_name,
      episode_id,
      filters,
      search_query_experimental: search_query,
    });

    if (offset !== undefined) {
      const hasMore = response.inferences.length > limit;
      return {
        inferences: response.inferences.slice(0, limit),
        hasNextPage: hasMore,
        hasPreviousPage: offset > 0,
      };
    }

    const {
      items: inferences,
      hasNextPage,
      hasPreviousPage,
    } = applyPaginationLogic(response.inferences, limit, { before, after });

    return {
      inferences,
      hasNextPage,
      hasPreviousPage,
    };
  } catch (error) {
    logger.error("Failed to list inferences:", error);
    if (isTensorZeroServerError(error)) {
      throw data("Error listing inferences", {
        status: error.status,
        statusText: error.statusText ?? undefined,
      });
    }
    throw data("Error listing inferences", { status: 500 });
  }
}

export async function countInferencesForFunction(
  function_name: string,
): Promise<number> {
  const client = getTensorZeroClient();
  const result = await client.getInferenceCount(function_name);
  return Number(result.inference_count);
}

export async function countInferencesForVariant(
  function_name: string,
  variant_name: string,
): Promise<number> {
  const client = getTensorZeroClient();
  const result = await client.getInferenceCount(function_name, {
    variantName: variant_name,
  });
  return Number(result.inference_count);
}

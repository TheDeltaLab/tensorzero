// Modified by Delta-AI under Apache 2.0
/**
 * URL helper functions that ensure proper encoding of identifiers.
 * Always use these instead of string interpolation to handle names with special characters.
 */

import type { ResolvedObject } from "~/types/tensorzero";

// ============================================================================
// Observability - Functions
// ============================================================================

export function toFunctionUrl(
  functionName: string,
  snapshotHash?: string,
): string {
  const base = `/observability/functions/${encodeURIComponent(functionName)}`;
  return appendSnapshotHash(base, snapshotHash);
}

export function toVariantUrl(
  functionName: string,
  variantName: string,
  snapshotHash?: string,
): string {
  const base = `/observability/functions/${encodeURIComponent(functionName)}/variants/${encodeURIComponent(variantName)}`;
  return appendSnapshotHash(base, snapshotHash);
}

// ============================================================================
// Observability - Inferences
// ============================================================================

export function toInferenceUrl(inferenceId: string): string {
  return `/observability/inferences/${encodeURIComponent(inferenceId)}`;
}

export function toInferencesListUrl(params?: { api_key?: string }): string {
  const search = new URLSearchParams();
  const apiKey = params?.api_key?.trim();
  if (apiKey) {
    search.set("api_key", apiKey);
  }
  const qs = search.toString();
  return qs ? `/observability/inferences?${qs}` : "/observability/inferences";
}

export function toInferenceApiUrl(inferenceId: string): string {
  return `/api/inference/${encodeURIComponent(inferenceId)}`;
}

// ============================================================================
// Observability - Episodes
// ============================================================================

export function toEpisodeUrl(episodeId: string): string {
  return `/observability/episodes/${encodeURIComponent(episodeId)}`;
}

// ============================================================================
// Resolved Object URLs
// ============================================================================

export function toResolvedObjectUrl(
  uuid: string,
  obj: ResolvedObject,
): string | null {
  switch (obj.type) {
    case "inference":
      return toInferenceUrl(uuid);
    case "episode":
      return toEpisodeUrl(uuid);
    case "chat_datapoint":
    case "json_datapoint":
    case "model_inference":
    case "boolean_feedback":
    case "float_feedback":
    case "comment_feedback":
    case "demonstration_feedback":
      return null;
    default: {
      const _exhaustiveCheck: never = obj;
      return _exhaustiveCheck;
    }
  }
}

// ============================================================================
// Helpers
// ============================================================================

function appendSnapshotHash(url: string, snapshotHash?: string): string {
  if (!snapshotHash) return url;
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}snapshot_hash=${encodeURIComponent(snapshotHash)}`;
}

// ============================================================================
// Internal API Routes
// ============================================================================

export function toResolveUuidApi(uuid: string): string {
  return `/api/tensorzero/resolve_uuid/${encodeURIComponent(uuid)}`;
}

export function toInferencePreviewApi(inferenceId: string): string {
  return `/api/tensorzero/inference_preview/${encodeURIComponent(inferenceId)}`;
}

export function toEpisodePreviewApi(episodeId: string): string {
  return `/api/tensorzero/episode_preview/${encodeURIComponent(episodeId)}`;
}

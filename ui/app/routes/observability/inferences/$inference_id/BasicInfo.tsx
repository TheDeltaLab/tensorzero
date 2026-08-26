// Modified by Delta-AI under Apache 2.0
import { Suspense } from "react";
import { Await } from "react-router";
import type { StoredInference } from "~/types/tensorzero";
import type { ParsedModelInferenceRow } from "~/utils/clickhouse/inference";
import { getTotalInferenceUsage } from "~/utils/clickhouse/helpers";
import {
  mergeUsage,
  usageFromTags,
  API_KEY_PUBLIC_ID_TAG,
} from "~/routes/observability/inferences/inferenceQuery";
import {
  firstFiniteMs,
  formatOutputTps,
  outputTpsExcludingTtft,
} from "~/utils/observability/usageDetails";
import {
  BasicInfoLayout,
  BasicInfoLayoutSkeleton,
  BasicInfoItem,
  BasicInfoItemTitle,
  BasicInfoItemContent,
} from "~/components/layout/BasicInfoLayout";
import Chip from "~/components/ui/Chip";
import {
  Timer,
  Calendar,
  InputIcon,
  Output,
  Cached,
  Cost,
} from "~/components/icons/Icons";
import {
  toFunctionUrl,
  toVariantUrl,
  toEpisodeUrl,
  toInferencesListUrl,
} from "~/utils/urls";
import { formatCost } from "~/utils/cost";
import { formatDateWithSeconds } from "~/utils/date";
import { TimestampTooltip } from "~/components/ui/TimestampTooltip";
import { getFunctionTypeIcon } from "~/utils/icon";
import { InlineAsyncError } from "~/components/ui/error/ErrorContentPrimitives";
import type { ModelInferencesData } from "./inference-data.server";
import {
  inferenceKindFromStored,
  isStandaloneInferenceKind,
} from "~/utils/observability/standaloneInference";

interface BasicInfoStreamingProps {
  inference: StoredInference;
  variantType: string;
  promise: Promise<ModelInferencesData>;
  locationKey: string;
}

export function BasicInfoStreaming({
  inference,
  variantType,
  promise,
  locationKey,
}: BasicInfoStreamingProps) {
  return (
    <Suspense key={locationKey} fallback={<BasicInfoLayoutSkeleton rows={6} />}>
      <Await
        resolve={promise}
        errorElement={
          <InlineAsyncError defaultMessage="Failed to load inference details" />
        }
      >
        {(modelInferences) => (
          <BasicInfo
            inference={inference}
            variantType={variantType}
            modelInferences={modelInferences}
          />
        )}
      </Await>
    </Suspense>
  );
}

interface BasicInfoProps {
  inference: StoredInference;
  variantType: string;
  modelInferences?: ParsedModelInferenceRow[];
}

export function BasicInfo({
  inference,
  variantType,
  modelInferences = [],
}: BasicInfoProps) {
  const snapshotHash = inference.snapshot_hash;
  const fromModels =
    modelInferences.length > 0
      ? getTotalInferenceUsage(modelInferences)
      : undefined;
  const inferenceUsage = mergeUsage(usageFromTags(inference.tags), fromModels);
  const ttftMs = firstFiniteMs([
    inference.ttft_ms,
    ...modelInferences.map((row) => row.ttft_ms),
  ]);
  const durationMs = firstFiniteMs([
    ...modelInferences.map((row) => row.response_time_ms),
    inference.processing_time_ms,
  ]);
  const outputTps = outputTpsExcludingTtft({
    outputTokens: inferenceUsage.output_tokens,
    durationMs,
    ttftMs,
  });
  const kind = inferenceKindFromStored(inference);
  const standalone = isStandaloneInferenceKind(kind);
  const apiKeyPublicId = inference.tags[API_KEY_PUBLIC_ID_TAG]?.trim();

  const functionIconConfig = getFunctionTypeIcon(kind);
  const hasCachedInferences = modelInferences.some((mi) => mi.cached);
  const allCached =
    modelInferences.length > 0 && modelInferences.every((mi) => mi.cached);
  const cacheStatus = allCached
    ? "FULL"
    : hasCachedInferences
      ? "PARTIAL"
      : "NONE";

  return (
    <BasicInfoLayout>
      <BasicInfoItem>
        <BasicInfoItemTitle>Function</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <Chip
            icon={functionIconConfig.icon}
            iconBg={functionIconConfig.iconBg}
            label={inference.function_name}
            secondaryLabel={`· ${kind}`}
            link={
              standalone
                ? undefined
                : toFunctionUrl(inference.function_name, snapshotHash)
            }
            font="mono"
          />
        </BasicInfoItemContent>
      </BasicInfoItem>

      <BasicInfoItem>
        <BasicInfoItemTitle>Variant</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <Chip
            label={inference.variant_name}
            secondaryLabel={`· ${variantType}`}
            link={
              standalone
                ? undefined
                : toVariantUrl(
                    inference.function_name,
                    inference.variant_name,
                    snapshotHash,
                  )
            }
            font="mono"
          />
        </BasicInfoItemContent>
      </BasicInfoItem>

      <BasicInfoItem>
        <BasicInfoItemTitle>Episode</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <Chip
            label={inference.episode_id}
            link={toEpisodeUrl(inference.episode_id)}
            font="mono"
          />
        </BasicInfoItemContent>
      </BasicInfoItem>

      <BasicInfoItem>
        <BasicInfoItemTitle>API Key</BasicInfoItemTitle>
        <BasicInfoItemContent>
          {apiKeyPublicId ? (
            <Chip
              label={apiKeyPublicId}
              link={toInferencesListUrl({ api_key: apiKeyPublicId })}
              font="mono"
              tooltip="Public id of the API key that made this request"
            />
          ) : (
            <span className="text-fg-muted">—</span>
          )}
        </BasicInfoItemContent>
      </BasicInfoItem>

      <BasicInfoItem>
        <BasicInfoItemTitle>Usage</BasicInfoItemTitle>
        <BasicInfoItemContent wrap>
          {inferenceUsage.input_tokens != null && (
            <Chip
              icon={<InputIcon className="text-fg-tertiary" />}
              label={`${inferenceUsage.input_tokens} in`}
              tooltip="Input tokens"
            />
          )}
          {inferenceUsage.output_tokens != null && (
            <Chip
              icon={<Output className="text-fg-tertiary" />}
              label={`${inferenceUsage.output_tokens} out`}
              tooltip="Output tokens"
            />
          )}
          {fromModels?.provider_cache_read_input_tokens != null &&
            fromModels.provider_cache_read_input_tokens > 0 && (
              <Chip
                icon={<Cached className="text-fg-tertiary" />}
                label={`${fromModels.provider_cache_read_input_tokens} cache read`}
                tooltip="Provider cache read tokens"
              />
            )}
          {fromModels?.provider_cache_write_input_tokens != null &&
            fromModels.provider_cache_write_input_tokens > 0 && (
              <Chip
                icon={<Cached className="text-fg-tertiary" />}
                label={`${fromModels.provider_cache_write_input_tokens} cache write`}
                tooltip="Provider cache write tokens"
              />
            )}
          {inferenceUsage.cost != null && (
            <Chip
              icon={<Cost className="text-fg-tertiary" />}
              label={formatCost(
                inferenceUsage.cost,
                inferenceUsage.currency ?? undefined,
              )}
              tooltip="Cost"
            />
          )}
          {inference.processing_time_ms != null && (
            <Chip
              icon={<Timer className="text-fg-tertiary" />}
              label={`${inference.processing_time_ms} ms`}
              tooltip="Processing time"
            />
          )}
          {ttftMs != null && (
            <Chip
              icon={<Timer className="text-fg-tertiary" />}
              label={`${ttftMs} ms TTFT`}
              tooltip="Time to first token"
            />
          )}
          {outputTps != null && (
            <Chip
              icon={<Output className="text-fg-tertiary" />}
              label={formatOutputTps(outputTps)}
              tooltip="Output tokens per second, excluding TTFT"
            />
          )}
          {(cacheStatus === "FULL" || cacheStatus === "PARTIAL") && (
            <Chip
              icon={<Cached className="text-fg-tertiary" />}
              label={cacheStatus === "FULL" ? "Cached" : "Partially Cached"}
              tooltip={
                cacheStatus === "FULL"
                  ? "All model inferences were cached by TensorZero"
                  : "Some model inferences were cached by TensorZero"
              }
            />
          )}
        </BasicInfoItemContent>
      </BasicInfoItem>

      <BasicInfoItem>
        <BasicInfoItemTitle>Timestamp</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <Chip
            icon={<Calendar className="text-fg-tertiary" />}
            label={formatDateWithSeconds(new Date(inference.timestamp))}
            tooltip={<TimestampTooltip timestamp={inference.timestamp} />}
          />
        </BasicInfoItemContent>
      </BasicInfoItem>
    </BasicInfoLayout>
  );
}

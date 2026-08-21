// Modified by Delta-AI under Apache 2.0
import { Suspense } from "react";
import { Await } from "react-router";
import { Skeleton } from "~/components/ui/skeleton";
import { SectionHeader, SectionLayout } from "~/components/layout/PageLayout";
import { SectionAsyncErrorState } from "~/components/ui/error/ErrorContentPrimitives";
import type { StoredInference } from "~/types/tensorzero";
import type { ParsedModelInferenceRow } from "~/utils/clickhouse/inference";
import {
  buildInferenceUsageDetails,
  buildModelInferenceUsageDetails,
  usageRecordToRows,
  type UsageDetailRow,
} from "~/utils/observability/usageDetails";

function UsageValue({ value }: { value: unknown }) {
  if (value === null || value === undefined) {
    return <span className="text-fg-muted">—</span>;
  }
  if (
    typeof value === "number" ||
    typeof value === "boolean" ||
    typeof value === "string"
  ) {
    return <span className="font-mono">{String(value)}</span>;
  }
  return (
    <pre className="bg-bg-tertiary overflow-x-auto rounded px-2 py-1 font-mono text-xs">
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}

function UsageDetailsTable({
  rows,
  title,
}: {
  rows: UsageDetailRow[];
  title?: string;
}) {
  if (rows.length === 0) {
    return null;
  }
  return (
    <div className="border-border overflow-hidden rounded-md border">
      {title && (
        <div className="bg-bg-secondary text-fg-secondary border-b px-3 py-2 text-sm font-medium">
          {title}
        </div>
      )}
      <table className="w-full text-sm">
        <tbody>
          {rows.map((row) => (
            <tr key={row.key} className="border-border border-b last:border-0">
              <td className="text-fg-secondary w-[38%] px-3 py-1.5 align-top font-mono text-xs">
                {row.label}
              </td>
              <td className="px-3 py-1.5 align-top">
                <UsageValue value={row.value} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function InferenceUsageDetails({
  inference,
  modelInferences,
}: {
  inference: Pick<StoredInference, "tags" | "processing_time_ms" | "ttft_ms">;
  modelInferences: ParsedModelInferenceRow[];
}) {
  const { rows, providerBlocks } = buildInferenceUsageDetails({
    tags: inference.tags,
    processingTimeMs: inference.processing_time_ms,
    ttftMs: inference.ttft_ms,
    modelInferences,
  });

  if (rows.length === 0 && providerBlocks.length === 0) {
    return <div className="text-fg-muted text-sm">No usage recorded</div>;
  }

  return (
    <div className="flex flex-col gap-3">
      <UsageDetailsTable rows={rows} />
      {providerBlocks.map((block) => (
        <UsageDetailsTable
          key={block.id}
          title={block.title}
          rows={usageRecordToRows(block.usage)}
        />
      ))}
    </div>
  );
}

export function ModelInferenceUsageDetails({
  inference,
}: {
  inference: ParsedModelInferenceRow;
}) {
  const { rows, providerBlocks } = buildModelInferenceUsageDetails(inference);
  if (rows.length === 0 && providerBlocks.length === 0) {
    return null;
  }
  return (
    <div className="flex flex-col gap-3">
      <UsageDetailsTable rows={rows} />
      {providerBlocks.map((block) => (
        <UsageDetailsTable
          key={block.id}
          title={block.title}
          rows={usageRecordToRows(block.usage)}
        />
      ))}
    </div>
  );
}

function UsageDetailsSkeleton() {
  return (
    <div className="border-border space-y-2 rounded-md border p-3">
      <Skeleton className="h-4 w-40" />
      <Skeleton className="h-4 w-56" />
      <Skeleton className="h-4 w-32" />
    </div>
  );
}

export function UsageDetailsSection({
  inference,
  promise,
  locationKey,
}: {
  inference: StoredInference;
  promise: Promise<ParsedModelInferenceRow[]>;
  locationKey: string;
}) {
  return (
    <SectionLayout>
      <SectionHeader heading="Usage details" />
      <Suspense
        key={`usage-${locationKey}`}
        fallback={<UsageDetailsSkeleton />}
      >
        <Await
          resolve={promise}
          errorElement={
            <SectionAsyncErrorState defaultMessage="Failed to load usage" />
          }
        >
          {(modelInferences) => (
            <InferenceUsageDetails
              inference={inference}
              modelInferences={modelInferences}
            />
          )}
        </Await>
      </Suspense>
    </SectionLayout>
  );
}

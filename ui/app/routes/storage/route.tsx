// Modified by Delta-AI under Apache 2.0
import type { Route } from "./+types/route";
import { Suspense, useEffect } from "react";
import {
  Await,
  data,
  useAsyncError,
  useFetcher,
  useLocation,
  type RouteHandle,
} from "react-router";
import { Database, HardDrive } from "lucide-react";
import {
  PageHeader,
  PageLayout,
  SectionHeader,
  SectionLayout,
  SectionsGroup,
} from "~/components/layout/PageLayout";
import { StatCard } from "~/components/analysis/StatCard";
import { Button } from "~/components/ui/button";
import { Card, CardContent } from "~/components/ui/card";
import { Input } from "~/components/ui/input";
import { Skeleton } from "~/components/ui/skeleton";
import { LayoutErrorBoundary, PageErrorContent } from "~/components/ui/error";
import { PostgresRequiredState } from "~/components/ui/PostgresRequiredState";
import { ReadOnlyGuard } from "~/components/utils/read-only-guard";
import { useReadOnly } from "~/context/read-only";
import { useToast } from "~/hooks/use-toast";
import { isPostgresAvailable } from "~/utils/postgres.server";
import { requireValidApiKeyIfEnabled } from "~/utils/auth.server";
import { isReadOnlyMode } from "~/utils/read-only.server";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { TensorZeroServerError } from "~/utils/tensorzero/errors";
import { logger } from "~/utils/logger";
import { formatBytes } from "~/utils/format";
import { formatCompactCount } from "~/routes/observability/analysis/analysisQuery";
import type { InferenceStorageStatsResponse } from "~/types/tensorzero";

export const handle: RouteHandle = {
  crumb: () => ["Storage"],
};

export async function loader(_args: Route.LoaderArgs) {
  if (!isPostgresAvailable()) {
    return {
      postgresAvailable: false as const,
      storageData: null,
    };
  }

  await requireValidApiKeyIfEnabled();

  return {
    postgresAvailable: true as const,
    storageData: getTensorZeroClient().getInferenceStorageStats(),
  };
}

export async function action({ request }: Route.ActionArgs) {
  if (isReadOnlyMode()) {
    return data(
      { error: "Retention policy cannot be changed in read-only mode." },
      { status: 403 },
    );
  }

  await requireValidApiKeyIfEnabled();

  const formData = await request.formData();

  const parseDays = (field: string): number | undefined | { error: string } => {
    const raw = formData.get(field);
    if (typeof raw !== "string" || raw.trim() === "") {
      return undefined;
    }
    const days = Number(raw);
    if (!Number.isInteger(days) || days < 1) {
      return { error: "Retention must be a whole number of days (1 or more)." };
    }
    return days;
  };

  const metadataDays = parseDays("metadata_retention_days");
  const dataDays = parseDays("data_retention_days");

  if (typeof metadataDays === "object") {
    return data({ error: metadataDays.error }, { status: 400 });
  }
  if (typeof dataDays === "object") {
    return data({ error: dataDays.error }, { status: 400 });
  }

  try {
    const retention = await getTensorZeroClient().updateInferenceRetention({
      metadata_retention_days: metadataDays,
      data_retention_days: dataDays,
    });
    return { success: true as const, retention };
  } catch (error) {
    logger.error("Failed to update inference retention", error);
    const message =
      error instanceof TensorZeroServerError
        ? error.message
        : "Failed to update retention policy. Please try again.";
    return data({ error: message }, { status: 400 });
  }
}

function StoragePageHeader() {
  return <PageHeader heading="Storage" />;
}

function StorageContentSkeleton() {
  return (
    <SectionsGroup>
      <SectionLayout>
        <SectionHeader heading="Inference Storage" />
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-28" />
          ))}
        </div>
      </SectionLayout>
      <SectionLayout>
        <SectionHeader heading="Retention Policy" />
        <Skeleton className="h-40" />
      </SectionLayout>
    </SectionsGroup>
  );
}

function StorageErrorState() {
  const error = useAsyncError();
  return <PageErrorContent error={error} />;
}

function formatTableName(name: string): string {
  return name
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function RetentionDaysInput({
  label,
  name,
  value,
  pinnedByToml,
  disabled,
}: {
  label: string;
  name: string;
  value?: number;
  pinnedByToml: boolean;
  disabled: boolean;
}) {
  return (
    <label className="space-y-2 text-sm font-medium">
      {label}
      <Input
        name={name}
        type="number"
        min={1}
        step={1}
        placeholder="Keep forever"
        defaultValue={value ?? ""}
        disabled={disabled}
        className="w-48"
      />
      {pinnedByToml && (
        <p className="text-muted-foreground text-xs font-normal">
          This value is set in tensorzero.toml and will overwrite dashboard
          changes on gateway restart.
        </p>
      )}
    </label>
  );
}

function RetentionPolicyForm({
  retention,
}: {
  retention: InferenceStorageStatsResponse["retention"];
}) {
  const fetcher = useFetcher<typeof action>();
  const isReadOnly = useReadOnly();
  const { toast } = useToast();
  const busy = fetcher.state !== "idle";

  useEffect(() => {
    if (fetcher.state === "idle" && fetcher.data && "success" in fetcher.data) {
      toast.success({ title: "Retention policy updated" });
    }
  }, [fetcher.state, fetcher.data, toast]);

  const error =
    fetcher.state === "idle" && fetcher.data && "error" in fetcher.data
      ? fetcher.data.error
      : null;

  return (
    <Card>
      <CardContent className="pt-6">
        <fetcher.Form
          key={`${retention.metadata_retention_days ?? ""}-${retention.data_retention_days ?? ""}`}
          method="post"
          className="flex flex-col gap-4"
        >
          <div className="flex flex-col gap-4 sm:flex-row sm:items-start">
            <RetentionDaysInput
              label="Metadata retention (days)"
              name="metadata_retention_days"
              value={retention.metadata_retention_days}
              pinnedByToml={retention.metadata_pinned_by_toml}
              disabled={busy || isReadOnly}
            />
            <RetentionDaysInput
              label="Payload retention (days)"
              name="data_retention_days"
              value={retention.data_retention_days}
              pinnedByToml={retention.data_pinned_by_toml}
              disabled={busy || isReadOnly}
            />
          </div>
          <p className="text-muted-foreground text-sm">
            Leave a field empty to keep data forever. Cleanup runs nightly via
            pg_cron partition drops (00:30 UTC metadata, 00:35 UTC payload) and
            does not block inference writes. Protected inferences are archived
            permanently and remain viewable on the inference detail page.
          </p>
          <div className="flex items-center gap-3">
            <ReadOnlyGuard>
              <Button type="submit" disabled={busy}>
                Save retention policy
              </Button>
            </ReadOnlyGuard>
            {error && <span className="text-destructive text-sm">{error}</span>}
          </div>
        </fetcher.Form>
      </CardContent>
    </Card>
  );
}

function StorageContent({ data }: { data: InferenceStorageStatsResponse }) {
  return (
    <SectionsGroup>
      <SectionLayout>
        <SectionHeader heading="Inference Storage" />
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {data.tables.map((table) => (
            <StatCard
              key={table.name}
              title={formatTableName(table.name)}
              value={formatBytes(Number(table.total_bytes))}
              icon={table.name.includes("archive") ? Database : HardDrive}
              description={`~${formatCompactCount(Number(table.estimated_rows))} rows · ${formatCompactCount(Number(table.partition_count))} partitions`}
            />
          ))}
        </div>
      </SectionLayout>
      <SectionLayout>
        <SectionHeader heading="Retention Policy" />
        <RetentionPolicyForm retention={data.retention} />
      </SectionLayout>
    </SectionsGroup>
  );
}

export default function StoragePage({ loaderData }: Route.ComponentProps) {
  const { postgresAvailable, storageData } = loaderData;
  const location = useLocation();

  if (!postgresAvailable) {
    return <PostgresRequiredState />;
  }

  return (
    <PageLayout>
      <StoragePageHeader />
      <Suspense key={location.key} fallback={<StorageContentSkeleton />}>
        <Await resolve={storageData} errorElement={<StorageErrorState />}>
          {(resolvedData) => <StorageContent data={resolvedData} />}
        </Await>
      </Suspense>
    </PageLayout>
  );
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  return <LayoutErrorBoundary error={error} />;
}

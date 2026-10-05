// Modified by Delta-AI under Apache 2.0
import type { Route } from "./+types/route";
import { Suspense, useCallback, useEffect, useState } from "react";
import {
  Await,
  data,
  useAsyncError,
  useFetcher,
  useLocation,
  useRevalidator,
  type RouteHandle,
} from "react-router";
import {
  ChevronDown,
  ChevronRight,
  Database,
  HardDrive,
  Pencil,
  Play,
  Plus,
  Trash,
} from "lucide-react";
import {
  PageHeader,
  PageLayout,
  SectionHeader,
  SectionLayout,
  SectionsGroup,
} from "~/components/layout/PageLayout";
import { StatCard } from "~/components/analysis/StatCard";
import { Badge } from "~/components/ui/badge";
import { Button } from "~/components/ui/button";
import { Card, CardContent } from "~/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "~/components/ui/dialog";
import { Input } from "~/components/ui/input";
import { Label } from "~/components/ui/label";
import { Skeleton } from "~/components/ui/skeleton";
import { Switch, SwitchSize } from "~/components/ui/switch";
import {
  Table,
  TableBody,
  TableCell,
  TableEmptyState,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import { TableItemTime } from "~/components/ui/TableItems";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
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
import type {
  CleanupRule,
  CleanupRun,
  InferenceStorageStatsResponse,
} from "~/types/tensorzero";

export const handle: RouteHandle = {
  crumb: () => ["Storage"],
};

interface StorageData {
  storageStats: InferenceStorageStatsResponse;
  cleanupRules: CleanupRule[];
  cleanupRuns: CleanupRun[];
}

export async function loader(_args: Route.LoaderArgs) {
  if (!isPostgresAvailable()) {
    return {
      postgresAvailable: false as const,
      storageData: null,
    };
  }

  await requireValidApiKeyIfEnabled();

  const client = getTensorZeroClient();
  return {
    postgresAvailable: true as const,
    storageData: Promise.all([
      client.getInferenceStorageStats(),
      client.getCleanupRules(),
      client.getCleanupRuns(),
    ]).then(
      ([storageStats, rulesResponse, runsResponse]): StorageData => ({
        storageStats,
        cleanupRules: rulesResponse.rules,
        cleanupRuns: runsResponse.runs,
      }),
    ),
  };
}

async function updateRetentionPolicy(formData: FormData) {
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

async function saveCleanupRule(
  intent: "create_rule" | "update_rule",
  formData: FormData,
) {
  const tagKeyRaw = formData.get("tag_key");
  const tagKey = typeof tagKeyRaw === "string" ? tagKeyRaw.trim() : "";
  if (tagKey === "") {
    return data({ error: "Tag key is required." }, { status: 400 });
  }

  const tagValueRaw = formData.get("tag_value");
  const tagValue =
    typeof tagValueRaw === "string" && tagValueRaw.trim() !== ""
      ? tagValueRaw.trim()
      : undefined;

  const olderThanDaysRaw = formData.get("older_than_days");
  const olderThanDays =
    typeof olderThanDaysRaw === "string" ? Number(olderThanDaysRaw) : NaN;
  if (!Number.isInteger(olderThanDays) || olderThanDays < 1) {
    return data(
      { error: "Older than must be a whole number of days (1 or more)." },
      { status: 400 },
    );
  }

  const enabled = formData.get("enabled") === "true";

  try {
    const client = getTensorZeroClient();
    if (intent === "create_rule") {
      await client.createCleanupRule({
        tag_key: tagKey,
        tag_value: tagValue,
        older_than_days: olderThanDays,
        enabled,
      });
      return { success: true as const, intent };
    }

    const ruleId = formData.get("rule_id");
    if (typeof ruleId !== "string" || ruleId === "") {
      return data({ error: "Rule ID is required." }, { status: 400 });
    }
    await client.updateCleanupRule(ruleId, {
      tag_key: tagKey,
      tag_value: tagValue,
      older_than_days: olderThanDays,
      enabled,
    });
    return { success: true as const, intent };
  } catch (error) {
    logger.error("Failed to save cleanup rule", error);
    const message =
      error instanceof TensorZeroServerError
        ? error.message
        : "Failed to save the cleanup rule. Please try again.";
    return data({ error: message }, { status: 400 });
  }
}

async function removeCleanupRule(formData: FormData) {
  const ruleId = formData.get("rule_id");
  if (typeof ruleId !== "string" || ruleId === "") {
    return data({ error: "Rule ID is required." }, { status: 400 });
  }
  try {
    await getTensorZeroClient().deleteCleanupRule(ruleId);
    return { success: true as const, intent: "delete_rule" as const };
  } catch (error) {
    logger.error("Failed to delete cleanup rule", error);
    const message =
      error instanceof TensorZeroServerError
        ? error.message
        : "Failed to delete the cleanup rule. Please try again.";
    return data({ error: message }, { status: 400 });
  }
}

async function runCleanupNow() {
  try {
    const response = await getTensorZeroClient().triggerCleanupRun();
    return {
      success: true as const,
      intent: "run_now" as const,
      cleanupEnabled: response.cleanup_enabled,
    };
  } catch (error) {
    logger.error("Failed to trigger cleanup run", error);
    const message =
      error instanceof TensorZeroServerError
        ? error.message
        : "Failed to trigger a cleanup run. Please try again.";
    return data({ error: message }, { status: 400 });
  }
}

export async function action({ request }: Route.ActionArgs) {
  if (isReadOnlyMode()) {
    return data(
      { error: "Storage settings cannot be changed in read-only mode." },
      { status: 403 },
    );
  }

  await requireValidApiKeyIfEnabled();

  const formData = await request.formData();
  const intent = formData.get("intent");

  if (intent === "create_rule" || intent === "update_rule") {
    return saveCleanupRule(intent, formData);
  }
  if (intent === "delete_rule") {
    return removeCleanupRule(formData);
  }
  if (intent === "run_now") {
    return runCleanupNow();
  }
  return updateRetentionPolicy(formData);
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
      <SectionLayout>
        <SectionHeader heading="Tag-Based Cleanup Rules" />
        <Skeleton className="h-40" />
      </SectionLayout>
      <SectionLayout>
        <SectionHeader heading="Cleanup Runs" />
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
          <input type="hidden" name="intent" value="save_retention" />
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

function CleanupRuleDialog({
  rule,
  open,
  onOpenChange,
}: {
  /** The rule being edited, or `null` to create a new one. */
  rule: CleanupRule | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const fetcher = useFetcher<typeof action>();
  const isReadOnly = useReadOnly();
  const { toast } = useToast();
  const [enabled, setEnabled] = useState(rule?.enabled ?? true);
  const [shouldCloseAfterSubmit, setShouldCloseAfterSubmit] = useState(false);
  const busy = fetcher.state !== "idle";

  useEffect(() => {
    if (!shouldCloseAfterSubmit || fetcher.state !== "idle" || !fetcher.data) {
      return;
    }
    setShouldCloseAfterSubmit(false);
    if ("error" in fetcher.data) {
      // Shown inline in the dialog.
      return;
    }
    toast.success({
      title: rule ? "Cleanup rule updated" : "Cleanup rule created",
    });
    onOpenChange(false);
  }, [
    shouldCloseAfterSubmit,
    fetcher.state,
    fetcher.data,
    toast,
    rule,
    onOpenChange,
  ]);

  const error =
    fetcher.state === "idle" && fetcher.data && "error" in fetcher.data
      ? fetcher.data.error
      : null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>
            {rule ? "Edit cleanup rule" : "New cleanup rule"}
          </DialogTitle>
          <DialogDescription>
            Delete inference payloads older than the given age for rows whose
            tag matches. Leave the tag value empty to match every row carrying
            the tag key.
          </DialogDescription>
        </DialogHeader>
        <fetcher.Form
          method="post"
          className="space-y-4"
          onSubmit={() => setShouldCloseAfterSubmit(true)}
        >
          <input
            type="hidden"
            name="intent"
            value={rule ? "update_rule" : "create_rule"}
          />
          {rule && <input type="hidden" name="rule_id" value={rule.id} />}
          <input
            type="hidden"
            name="enabled"
            value={enabled ? "true" : "false"}
          />
          <div className="space-y-2">
            <Label htmlFor="tag_key">Tag key</Label>
            <Input
              id="tag_key"
              name="tag_key"
              required
              defaultValue={rule?.tag_key ?? ""}
              placeholder="e.g. environment"
              disabled={busy}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="tag_value">
              Tag value{" "}
              <span className="text-muted-foreground">(optional)</span>
            </Label>
            <Input
              id="tag_value"
              name="tag_value"
              defaultValue={rule?.tag_value ?? ""}
              placeholder="Any value"
              disabled={busy}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="older_than_days">
              Delete rows older than (days)
            </Label>
            <Input
              id="older_than_days"
              name="older_than_days"
              type="number"
              min={1}
              step={1}
              required
              defaultValue={rule?.older_than_days ?? 30}
              className="w-32"
              disabled={busy}
            />
          </div>
          <label className="flex items-center gap-2 text-sm font-medium">
            <Switch
              size={SwitchSize.Small}
              checked={enabled}
              onCheckedChange={setEnabled}
              disabled={busy || isReadOnly}
            />
            Enabled
          </label>
          {error && <p className="text-destructive text-sm">{error}</p>}
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
              disabled={busy}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={busy || isReadOnly}>
              {rule ? "Save rule" : "Create rule"}
            </Button>
          </DialogFooter>
        </fetcher.Form>
      </DialogContent>
    </Dialog>
  );
}

function CleanupRulesSection({
  rules,
  onRunTriggered,
}: {
  rules: CleanupRule[];
  onRunTriggered: () => void;
}) {
  const { toast } = useToast();
  const deleteFetcher = useFetcher<typeof action>();
  const runFetcher = useFetcher<typeof action>();
  const [dialogOpen, setDialogOpen] = useState(false);
  // Remount the dialog per target so its fetcher and field state start clean.
  const [dialogKey, setDialogKey] = useState(0);
  const [editingRule, setEditingRule] = useState<CleanupRule | null>(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [ruleToDelete, setRuleToDelete] = useState<CleanupRule | null>(null);
  const [pendingDeleteToast, setPendingDeleteToast] = useState(false);
  const [pendingRunToast, setPendingRunToast] = useState(false);

  useEffect(() => {
    if (
      !pendingDeleteToast ||
      deleteFetcher.state !== "idle" ||
      !deleteFetcher.data
    ) {
      return;
    }
    setPendingDeleteToast(false);
    if ("error" in deleteFetcher.data) {
      toast.error({
        title: "Failed to delete cleanup rule",
        description: deleteFetcher.data.error,
      });
      return;
    }
    toast.success({ title: "Cleanup rule deleted" });
  }, [pendingDeleteToast, deleteFetcher.state, deleteFetcher.data, toast]);

  useEffect(() => {
    if (!pendingRunToast || runFetcher.state !== "idle" || !runFetcher.data) {
      return;
    }
    setPendingRunToast(false);
    if ("error" in runFetcher.data) {
      toast.error({
        title: "Failed to trigger cleanup",
        description: runFetcher.data.error,
      });
      return;
    }
    if (
      !("intent" in runFetcher.data) ||
      runFetcher.data.intent !== "run_now"
    ) {
      return;
    }
    if (!runFetcher.data.cleanupEnabled) {
      toast.error({
        title: "Cleanup is disabled",
        description:
          "Tag-based cleanup is disabled in the gateway config ([gateway.cleanup].enabled = false), so no run was started.",
      });
      return;
    }
    toast.success({ title: "Cleanup run triggered" });
    onRunTriggered();
  }, [
    pendingRunToast,
    runFetcher.state,
    runFetcher.data,
    toast,
    onRunTriggered,
  ]);

  const openCreateDialog = () => {
    setEditingRule(null);
    setDialogKey((key) => key + 1);
    setDialogOpen(true);
  };

  const openEditDialog = (rule: CleanupRule) => {
    setEditingRule(rule);
    setDialogKey((key) => key + 1);
    setDialogOpen(true);
  };

  const handleDeleteClick = (rule: CleanupRule) => {
    setRuleToDelete(rule);
    setDeleteDialogOpen(true);
  };

  const confirmDelete = () => {
    if (ruleToDelete) {
      setPendingDeleteToast(true);
      deleteFetcher.submit(
        { intent: "delete_rule", rule_id: ruleToDelete.id },
        { method: "post" },
      );
    }
    setDeleteDialogOpen(false);
    setRuleToDelete(null);
  };

  return (
    <SectionLayout>
      <div className="flex flex-wrap items-center justify-between gap-4">
        <SectionHeader heading="Tag-Based Cleanup Rules" />
        <div className="flex items-center gap-2">
          <ReadOnlyGuard asChild>
            <Button
              variant="outline"
              size="sm"
              disabled={runFetcher.state !== "idle"}
              onClick={() => {
                setPendingRunToast(true);
                runFetcher.submit({ intent: "run_now" }, { method: "post" });
              }}
            >
              <Play className="mr-2 h-4 w-4" />
              Run cleanup now
            </Button>
          </ReadOnlyGuard>
          <ReadOnlyGuard asChild>
            <Button size="sm" onClick={openCreateDialog}>
              <Plus className="mr-2 h-4 w-4" />
              New rule
            </Button>
          </ReadOnlyGuard>
        </div>
      </div>
      <p className="text-muted-foreground text-sm">
        Rules delete inference payloads older than the given age for rows
        carrying a matching tag. The cleanup worker applies enabled rules
        periodically; progress appears in the Cleanup Runs section below.
      </p>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Tag Key</TableHead>
            <TableHead>Tag Value</TableHead>
            <TableHead className="w-36">Older Than</TableHead>
            <TableHead className="w-28">Status</TableHead>
            <TableHead className="w-52 whitespace-nowrap">Created</TableHead>
            <TableHead className="w-24"></TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rules.length === 0 ? (
            <TableEmptyState message="No cleanup rules configured" />
          ) : (
            rules.map((rule) => (
              <TableRow key={rule.id}>
                <TableCell>
                  <code className="font-mono text-sm">{rule.tag_key}</code>
                </TableCell>
                <TableCell>
                  {rule.tag_value ? (
                    <code className="font-mono text-sm">{rule.tag_value}</code>
                  ) : (
                    <span className="text-fg-muted italic">any value</span>
                  )}
                </TableCell>
                <TableCell>
                  {rule.older_than_days}{" "}
                  {rule.older_than_days === 1 ? "day" : "days"}
                </TableCell>
                <TableCell>
                  <Badge variant={rule.enabled ? "default" : "outline"}>
                    {rule.enabled ? "Enabled" : "Disabled"}
                  </Badge>
                </TableCell>
                <TableCell>
                  <TableItemTime timestamp={rule.created_at} />
                </TableCell>
                <TableCell>
                  <div className="flex items-center justify-end gap-2">
                    <ReadOnlyGuard asChild>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => openEditDialog(rule)}
                        aria-label={`Edit rule for tag ${rule.tag_key}`}
                        className="opacity-60 transition-opacity hover:opacity-100"
                      >
                        <Pencil className="h-4 w-4" />
                      </Button>
                    </ReadOnlyGuard>
                    <ReadOnlyGuard asChild>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => handleDeleteClick(rule)}
                        aria-label={`Delete rule for tag ${rule.tag_key}`}
                        className="opacity-60 transition-opacity hover:opacity-100"
                      >
                        <Trash className="h-4 w-4" />
                      </Button>
                    </ReadOnlyGuard>
                  </div>
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>

      <CleanupRuleDialog
        key={dialogKey}
        rule={editingRule}
        open={dialogOpen}
        onOpenChange={setDialogOpen}
      />

      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              Are you sure you want to delete the cleanup rule for tag{" "}
              <span className="font-mono text-lg font-bold text-red-500">
                {ruleToDelete?.tag_key}
                {ruleToDelete?.tag_value ? `=${ruleToDelete.tag_value}` : ""}
              </span>
              ?
            </DialogTitle>
            <DialogDescription>
              Future cleanup passes will no longer delete rows matching this
              rule. This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="flex justify-between gap-2">
            <Button
              variant="outline"
              onClick={() => setDeleteDialogOpen(false)}
            >
              Cancel
            </Button>
            <div className="flex-1" />
            <Button variant="destructive" onClick={confirmDelete}>
              <Trash className="inline h-4 w-4" />
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </SectionLayout>
  );
}

function runStatusBadgeVariant(status: string) {
  switch (status) {
    case "running":
      return "secondary" as const;
    case "completed":
      return "default" as const;
    case "failed":
      return "destructive" as const;
    default:
      return "outline" as const;
  }
}

function stepStatusBadgeVariant(status: string) {
  switch (status) {
    case "pending":
      return "outline" as const;
    case "running":
      return "secondary" as const;
    case "done":
      return "default" as const;
    case "failed":
      return "destructive" as const;
    default:
      return "outline" as const;
  }
}

function totalRowsDeleted(run: CleanupRun): number {
  return run.steps.reduce(
    (total, step) => total + Number(step.rows_deleted),
    0,
  );
}

function TruncatedError({ error }: { error?: string }) {
  if (!error) {
    return <span className="text-fg-muted">—</span>;
  }
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="block max-w-64 truncate">{error}</span>
      </TooltipTrigger>
      <TooltipContent>
        <p className="max-w-md whitespace-pre-wrap">{error}</p>
      </TooltipContent>
    </Tooltip>
  );
}

const RUNS_TABLE_COLUMN_COUNT = 7;

function CleanupRunRow({
  run,
  expanded,
  onToggle,
}: {
  run: CleanupRun;
  expanded: boolean;
  onToggle: () => void;
}) {
  return (
    <>
      <TableRow>
        <TableCell className="w-10">
          <Button
            variant="ghost"
            size="icon"
            onClick={onToggle}
            aria-expanded={expanded}
            aria-label={expanded ? "Hide steps" : "Show steps"}
            className="opacity-60 transition-opacity hover:opacity-100"
          >
            {expanded ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            )}
          </Button>
        </TableCell>
        <TableCell className="w-52 whitespace-nowrap">
          <TableItemTime timestamp={run.started_at} />
        </TableCell>
        <TableCell className="w-28">
          <Badge variant="outline">{run.trigger}</Badge>
        </TableCell>
        <TableCell className="w-28">
          <Badge variant={runStatusBadgeVariant(run.status)}>
            {run.status}
          </Badge>
        </TableCell>
        <TableCell className="w-52 whitespace-nowrap">
          {run.finished_at ? (
            <TableItemTime timestamp={run.finished_at} />
          ) : (
            <span className="text-fg-muted">—</span>
          )}
        </TableCell>
        <TableCell className="w-28">
          {formatCompactCount(totalRowsDeleted(run))}
        </TableCell>
        <TableCell>
          <TruncatedError error={run.error} />
        </TableCell>
      </TableRow>
      {expanded && (
        <TableRow>
          <TableCell
            colSpan={RUNS_TABLE_COLUMN_COUNT}
            className="bg-bg-secondary"
          >
            {run.steps.length === 0 ? (
              <p className="text-muted-foreground text-sm">
                No steps recorded yet.
              </p>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Table</TableHead>
                    <TableHead className="w-28">Status</TableHead>
                    <TableHead className="w-32">Rows Deleted</TableHead>
                    <TableHead className="w-52 whitespace-nowrap">
                      Started
                    </TableHead>
                    <TableHead className="w-52 whitespace-nowrap">
                      Finished
                    </TableHead>
                    <TableHead>Error</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {run.steps.map((step) => (
                    <TableRow key={step.id}>
                      <TableCell>
                        <code className="font-mono text-sm">
                          {step.table_name}
                        </code>
                      </TableCell>
                      <TableCell>
                        <Badge variant={stepStatusBadgeVariant(step.status)}>
                          {step.status}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        {formatCompactCount(Number(step.rows_deleted))}
                        {" / "}
                        {step.total_rows !== undefined
                          ? formatCompactCount(Number(step.total_rows))
                          : "?"}
                      </TableCell>
                      <TableCell>
                        <TableItemTime timestamp={step.started_at} />
                      </TableCell>
                      <TableCell>
                        {step.finished_at ? (
                          <TableItemTime timestamp={step.finished_at} />
                        ) : (
                          <span className="text-fg-muted">—</span>
                        )}
                      </TableCell>
                      <TableCell>
                        <TruncatedError error={step.error} />
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </TableCell>
        </TableRow>
      )}
    </>
  );
}

function CleanupRunsSection({
  runs,
  runTriggeredAt,
}: {
  runs: CleanupRun[];
  /** Timestamp of the last successful manual trigger, if any. */
  runTriggeredAt: number | null;
}) {
  const revalidator = useRevalidator();
  const [expandedRunIds, setExpandedRunIds] = useState<ReadonlySet<string>>(
    new Set(),
  );

  const hasRunningRun = runs.some((run) => run.status === "running");
  // A triggered run takes a moment to appear in the history, so keep polling
  // briefly after a manual trigger even before any run shows as `running`.
  const awaitingTriggeredRun =
    runTriggeredAt !== null && Date.now() - runTriggeredAt < 15_000;

  useEffect(() => {
    if (!hasRunningRun && !awaitingTriggeredRun) {
      return;
    }
    const interval = setInterval(() => revalidator.revalidate(), 2000);
    return () => clearInterval(interval);
  }, [hasRunningRun, awaitingTriggeredRun, revalidator]);

  const toggleExpanded = (runId: string) => {
    setExpandedRunIds((current) => {
      const next = new Set(current);
      if (next.has(runId)) {
        next.delete(runId);
      } else {
        next.add(runId);
      }
      return next;
    });
  };

  return (
    <SectionLayout>
      <SectionHeader heading="Cleanup Runs" />
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-10"></TableHead>
            <TableHead className="w-52 whitespace-nowrap">Started</TableHead>
            <TableHead className="w-28">Trigger</TableHead>
            <TableHead className="w-28">Status</TableHead>
            <TableHead className="w-52 whitespace-nowrap">Finished</TableHead>
            <TableHead className="w-28">Rows Deleted</TableHead>
            <TableHead>Error</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {runs.length === 0 ? (
            <TableEmptyState message="No cleanup runs recorded yet" />
          ) : (
            runs.map((run) => (
              <CleanupRunRow
                key={run.id}
                run={run}
                expanded={expandedRunIds.has(run.id)}
                onToggle={() => toggleExpanded(run.id)}
              />
            ))
          )}
        </TableBody>
      </Table>
    </SectionLayout>
  );
}

function StorageContent({ data }: { data: StorageData }) {
  const { storageStats, cleanupRules, cleanupRuns } = data;
  const [runTriggeredAt, setRunTriggeredAt] = useState<number | null>(null);
  const handleRunTriggered = useCallback(() => {
    setRunTriggeredAt(Date.now());
  }, []);

  return (
    <SectionsGroup>
      <SectionLayout>
        <SectionHeader heading="Inference Storage" />
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {storageStats.tables.map((table) => (
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
        <RetentionPolicyForm retention={storageStats.retention} />
      </SectionLayout>
      <CleanupRulesSection
        rules={cleanupRules}
        onRunTriggered={handleRunTriggered}
      />
      <CleanupRunsSection runs={cleanupRuns} runTriggeredAt={runTriggeredAt} />
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

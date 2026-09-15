// Modified by Delta-AI under Apache 2.0
import type { Route } from "./+types/route";
import { data, Link } from "react-router";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import { LayoutErrorBoundary } from "~/components/ui/error/LayoutErrorBoundary";
import { Badge } from "~/components/ui/badge";
import { Code } from "~/components/ui/code";
import { CodeEditor } from "~/components/ui/code-editor";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { TensorZeroServerError } from "~/utils/tensorzero/errors";
import { formatDateWithSeconds } from "~/utils/date";
import { formatDurationMs, statusBadgeVariant } from "../AsyncTasksTable";

export async function loader({ params }: Route.LoaderArgs) {
  const taskId = params.taskId;
  const client = getTensorZeroClient();
  try {
    const task = await client.getAsyncTask(taskId);
    return { task };
  } catch (error) {
    if (error instanceof TensorZeroServerError && error.status === 404) {
      throw data(`Async task ${taskId} not found`, { status: 404 });
    }
    throw error;
  }
}

function PropertyRow({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-fg-muted text-sm">{label}</span>
      <span className="text-sm">{children}</span>
    </div>
  );
}

/// Inference id from a completed task's response payload: a bare UUID for
/// chat/responses, `msg_`-prefixed for messages. Mirrors the backend's
/// `extract_inference_id` in `endpoints/internal/async_tasks.rs`.
function extractInferenceId(response: unknown): string | undefined {
  if (typeof response !== "object" || response === null) {
    return undefined;
  }
  const raw = (response as Record<string, unknown>).id;
  if (typeof raw !== "string") {
    return undefined;
  }
  const id = raw.startsWith("msg_") ? raw.slice("msg_".length) : raw;
  if (id === "") {
    return undefined;
  }
  return id;
}

export default function AsyncTaskDetailPage({
  loaderData,
}: Route.ComponentProps) {
  const { task } = loaderData;
  const inferenceId =
    task.status === "completed" ? extractInferenceId(task.response) : undefined;

  return (
    <PageLayout>
      <PageHeader heading="Async Task" />
      <SectionLayout>
        <div className="flex flex-col gap-6">
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <PropertyRow label="Task ID">
              <Code>{task.task_id}</Code>
            </PropertyRow>
            <PropertyRow label="Status">
              <Badge variant={statusBadgeVariant(task.status)}>
                {task.status}
              </Badge>
            </PropertyRow>
            {inferenceId && (
              <PropertyRow label="Inference">
                <Link
                  to={`/observability/inferences/${inferenceId}`}
                  className="hover:underline"
                >
                  <Code>{inferenceId}</Code>
                </Link>
              </PropertyRow>
            )}
            {task.status === "queued" && (
              <PropertyRow label="Queue Position">
                {task.queue_position !== undefined
                  ? Number(task.queue_position)
                  : "—"}
              </PropertyRow>
            )}
            {task.status === "running" && (
              <>
                <PropertyRow label="Started">
                  {task.started_at
                    ? formatDateWithSeconds(new Date(task.started_at))
                    : "—"}
                </PropertyRow>
                <PropertyRow label="Elapsed">
                  {task.elapsed_ms !== undefined
                    ? formatDurationMs(Number(task.elapsed_ms))
                    : "—"}
                </PropertyRow>
              </>
            )}
          </div>

          {task.status === "completed" && (
            <div className="flex flex-col gap-2">
              <h3 className="font-medium">Response</h3>
              <CodeEditor
                allowedLanguages={["json"]}
                value={JSON.stringify(task.response, null, 2)}
                readOnly
              />
            </div>
          )}

          {(task.status === "failed" || task.status === "cancelled") && (
            <div className="flex flex-col gap-2">
              <h3 className="font-medium">Error</h3>
              <CodeEditor
                allowedLanguages={["json"]}
                value={JSON.stringify(task.error ?? null, null, 2)}
                readOnly
              />
            </div>
          )}

          <div className="flex flex-col gap-2">
            <h3 className="font-medium">Endpoints</h3>
            <div className="text-fg-muted flex flex-col gap-1 text-sm">
              <span>
                <Code>GET /v1/async_tasks/{task.task_id}</Code>
              </span>
              <span>
                <Code>GET /v1/async_tasks/{task.task_id}/stream</Code>
              </span>
            </div>
          </div>
        </div>
      </SectionLayout>
    </PageLayout>
  );
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  return <LayoutErrorBoundary error={error} />;
}

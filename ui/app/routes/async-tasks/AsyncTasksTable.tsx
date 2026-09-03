// Modified by Delta-AI under Apache 2.0
import type { AsyncTaskStatus, AsyncTaskSummary } from "~/types/tensorzero";
import { Badge } from "~/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableEmptyState,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import { TableItemShortUuid, TableItemTime } from "~/components/ui/TableItems";

const STATUS_BADGE_VARIANTS = {
  queued: "outline",
  running: "secondary",
  completed: "default",
  failed: "destructive",
  cancelled: "warning",
} as const;

export function statusBadgeVariant(status: AsyncTaskStatus) {
  return STATUS_BADGE_VARIANTS[status];
}

export function formatDurationMs(durationMs: number): string {
  if (durationMs < 1000) {
    return `${durationMs}ms`;
  }
  const seconds = durationMs / 1000;
  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.round(seconds % 60);
  return `${minutes}m ${remainingSeconds}s`;
}

export const ASYNC_TASKS_TABLE_COLUMN_COUNT = 6;

export function AsyncTasksTableRows({ tasks }: { tasks: AsyncTaskSummary[] }) {
  if (tasks.length === 0) {
    return <TableEmptyState message="No async tasks found" />;
  }

  return (
    <>
      {tasks.map((task) => (
        <TableRow key={task.task_id}>
          <TableCell>
            <TableItemShortUuid
              id={task.task_id}
              link={`/async-tasks/${task.task_id}`}
            />
          </TableCell>
          <TableCell>{task.api_kind ?? "—"}</TableCell>
          <TableCell className="max-w-xs">
            <span className="block truncate">{task.model ?? "—"}</span>
          </TableCell>
          <TableCell>
            <Badge variant={statusBadgeVariant(task.status)}>
              {task.status}
            </Badge>
          </TableCell>
          <TableCell className="whitespace-nowrap">
            {task.duration_ms !== undefined
              ? formatDurationMs(Number(task.duration_ms))
              : "—"}
          </TableCell>
          <TableCell className="w-52 whitespace-nowrap">
            <TableItemTime timestamp={task.enqueue_at} />
          </TableCell>
        </TableRow>
      ))}
    </>
  );
}

export default function AsyncTasksTable({
  tasks,
}: {
  tasks: AsyncTaskSummary[];
}) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead className="w-36">Task ID</TableHead>
          <TableHead className="w-28">API</TableHead>
          <TableHead>Model</TableHead>
          <TableHead className="w-28">Status</TableHead>
          <TableHead className="w-28">Duration</TableHead>
          <TableHead className="w-52 whitespace-nowrap">Enqueued</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        <AsyncTasksTableRows tasks={tasks} />
      </TableBody>
    </Table>
  );
}

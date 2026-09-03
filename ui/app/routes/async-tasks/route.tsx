// Modified by Delta-AI under Apache 2.0
import { Suspense } from "react";
import type { Route } from "./+types/route";
import { Await, data, useLocation, useNavigate } from "react-router";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import { ActionBar } from "~/components/layout/ActionBar";
import PageButtons from "~/components/utils/PageButtons";
import { LayoutErrorBoundary } from "~/components/ui/error/LayoutErrorBoundary";
import {
  AsyncTasksTableRows,
  ASYNC_TASKS_TABLE_COLUMN_COUNT,
} from "./AsyncTasksTable";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import type { AsyncTaskStatus, AsyncTaskSummary } from "~/types/tensorzero";
import { Skeleton } from "~/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";
import {
  Table,
  TableAsyncErrorState,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";

const MAX_PAGE_SIZE = 50;
const DEFAULT_PAGE_SIZE = 20;

const ASYNC_TASK_STATUSES: AsyncTaskStatus[] = [
  "queued",
  "running",
  "completed",
  "failed",
  "cancelled",
];

export type AsyncTasksData = {
  tasks: AsyncTaskSummary[];
  hasMore: boolean;
};

function parseInteger(value: string | null, fallback: number) {
  if (!value) return fallback;
  const parsed = Number.parseInt(value, 10);
  return Number.isNaN(parsed) ? fallback : parsed;
}

function parseStatus(value: string | null): AsyncTaskStatus | undefined {
  if (!value) return undefined;
  return ASYNC_TASK_STATUSES.includes(value as AsyncTaskStatus)
    ? (value as AsyncTaskStatus)
    : undefined;
}

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const limitParam = parseInteger(
    url.searchParams.get("limit"),
    DEFAULT_PAGE_SIZE,
  );
  const offsetParam = parseInteger(url.searchParams.get("offset"), 0);
  const status = parseStatus(url.searchParams.get("status"));
  const limit = Math.max(1, limitParam);
  const offset = Math.max(0, offsetParam);

  if (limit > MAX_PAGE_SIZE) {
    throw data(`Limit cannot exceed ${MAX_PAGE_SIZE}`, { status: 400 });
  }

  const client = getTensorZeroClient();

  // Return promise WITHOUT awaiting - enables streaming/skeleton loading
  const tasksDataPromise = client
    .listAsyncTasks({
      limit: limit + 1,
      offset,
      status,
    })
    .then((response) => {
      const hasMore = response.tasks.length > limit;
      const tasks = response.tasks.slice(0, limit);
      return { tasks, hasMore };
    });

  return {
    tasksData: tasksDataPromise,
    offset,
    limit,
    status,
  };
}

// Skeleton rows for loading state - matches table columns
function SkeletonRows() {
  return (
    <>
      {Array.from({ length: 10 }).map((_, i) => (
        <TableRow key={i}>
          <TableCell>
            <Skeleton className="h-5 w-24" />
          </TableCell>
          <TableCell>
            <Skeleton className="h-5 w-16" />
          </TableCell>
          <TableCell className="max-w-xs">
            <Skeleton className="h-5 w-40" />
          </TableCell>
          <TableCell>
            <Skeleton className="h-5 w-20" />
          </TableCell>
          <TableCell>
            <Skeleton className="h-5 w-14" />
          </TableCell>
          <TableCell className="w-52 whitespace-nowrap">
            <Skeleton className="h-5 w-36" />
          </TableCell>
        </TableRow>
      ))}
    </>
  );
}

export default function AsyncTasksPage({ loaderData }: Route.ComponentProps) {
  const navigate = useNavigate();
  const location = useLocation();
  const { tasksData, offset, limit, status } = loaderData;

  const updateOffset = (nextOffset: number) => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("offset", String(nextOffset));
    searchParams.set("limit", String(limit));
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  const handleNextPage = () => {
    updateOffset(offset + limit);
  };

  const handlePreviousPage = () => {
    updateOffset(Math.max(0, offset - limit));
  };

  const handleStatusChange = (value: string) => {
    const searchParams = new URLSearchParams(window.location.search);
    if (value === "all") {
      searchParams.delete("status");
    } else {
      searchParams.set("status", value);
    }
    searchParams.set("offset", "0");
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  return (
    <PageLayout>
      <PageHeader heading="Async Tasks" />
      <SectionLayout>
        <ActionBar>
          <Select value={status ?? "all"} onValueChange={handleStatusChange}>
            <SelectTrigger className="w-40" aria-label="Filter by status">
              <SelectValue placeholder="Status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All statuses</SelectItem>
              {ASYNC_TASK_STATUSES.map((s) => (
                <SelectItem key={s} value={s}>
                  {s}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </ActionBar>
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
            <Suspense key={location.search} fallback={<SkeletonRows />}>
              <Await
                resolve={tasksData}
                errorElement={
                  <TableAsyncErrorState
                    colSpan={ASYNC_TASKS_TABLE_COLUMN_COUNT}
                    defaultMessage="Failed to load async tasks"
                  />
                }
              >
                {({ tasks }) => <AsyncTasksTableRows tasks={tasks} />}
              </Await>
            </Suspense>
          </TableBody>
        </Table>
        <Suspense key={location.search} fallback={<PageButtons disabled />}>
          <Await resolve={tasksData} errorElement={<PageButtons disabled />}>
            {({ hasMore }) => (
              <PageButtons
                onPreviousPage={handlePreviousPage}
                onNextPage={handleNextPage}
                disablePrevious={offset <= 0}
                disableNext={!hasMore}
              />
            )}
          </Await>
        </Suspense>
      </SectionLayout>
    </PageLayout>
  );
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  return <LayoutErrorBoundary error={error} />;
}

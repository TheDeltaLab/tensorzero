// Modified by Delta-AI under Apache 2.0
import type { InferenceMetadata } from "~/types/tensorzero";
import { uuidv7ToTimestamp } from "~/utils/clickhouse/helpers";
import {
  Table,
  TableAsyncErrorState,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import { TableItemTime } from "~/components/ui/TableItems";
import { toInferenceUrl, toEpisodeUrl, toFunctionUrl } from "~/utils/urls";
import { Button } from "~/components/ui/button";
import { Badge } from "~/components/ui/badge";
import { Eye, Layers, ShieldCheck } from "lucide-react";
import { Suspense, type ReactNode } from "react";
import {
  Link,
  useNavigate,
  useLocation,
  useSearchParams,
  Await,
} from "react-router";
import { Skeleton } from "~/components/ui/skeleton";
import PageButtons from "~/components/utils/PageButtons";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { getFunctionTypeIcon } from "~/utils/icon";
import { formatCost } from "~/utils/cost";
import {
  cachedFromTags,
  fallbackCountFromTags,
  providerFromTags,
  statusCodeFromTags,
} from "./inferenceQuery";
import {
  isStandaloneFunctionName,
  observabilityInferenceKind,
} from "~/utils/observability/standaloneInference";
import { splitInferenceTags } from "~/utils/observability/inferenceTags";
import {
  formatOutputTps,
  outputTpsExcludingTtft,
  toFiniteMs,
} from "~/utils/observability/usageDetails";

export type InferenceListRow = InferenceMetadata & {
  tags?: Record<string, string>;
  processing_time_ms?: number;
  ttft_ms?: number;
  input_tokens?: number;
  output_tokens?: number;
  cost?: number | null;
  currency?: string | null;
};

export type InferencesData = {
  inferences: InferenceListRow[];
  hasNextPage: boolean;
  hasPreviousPage: boolean;
  /** inference id → protected_at, for inferences protected from cleanup */
  protection: Record<string, string>;
};

const COLUMN_COUNT = 14;

function SkeletonRows() {
  return (
    <>
      {Array.from({ length: 10 }).map((_, i) => (
        <TableRow key={i}>
          <TableCell>
            <Skeleton className="h-5 w-36" />
          </TableCell>
          <TableCell>
            <Skeleton className="h-5 w-16" />
          </TableCell>
          <TableCell>
            <Skeleton className="h-5 w-28" />
          </TableCell>
          <TableCell>
            <Skeleton className="ml-auto h-5 w-12" />
          </TableCell>
          <TableCell>
            <Skeleton className="ml-auto h-5 w-14" />
          </TableCell>
          <TableCell>
            <Skeleton className="ml-auto h-5 w-14" />
          </TableCell>
          <TableCell>
            <Skeleton className="ml-auto h-5 w-12" />
          </TableCell>
          <TableCell>
            <Skeleton className="ml-auto h-5 w-16" />
          </TableCell>
          <TableCell>
            <Skeleton className="h-5 w-14" />
          </TableCell>
          <TableCell>
            <Skeleton className="h-5 w-10" />
          </TableCell>
          <TableCell>
            <Skeleton className="h-5 w-16" />
          </TableCell>
          <TableCell className="w-[44px]" />
          <TableCell className="w-[36px]" />
          <TableCell className="w-[36px]" />
        </TableRow>
      ))}
    </>
  );
}

function TableRows({ data }: { data: InferencesData }) {
  const navigate = useNavigate();
  const { inferences } = data;

  if (inferences.length === 0) {
    return <TableEmptyState message="No inferences found" />;
  }

  return (
    <>
      {inferences.map((inference) => {
        const kind = observabilityInferenceKind({
          functionName: inference.function_name,
          functionType: inference.function_type,
          tags: inference.tags,
        });
        const provider = providerFromTags(
          inference.tags,
          inference.variant_name,
        );
        const fallbackCount = fallbackCountFromTags(inference.tags);
        const cached = cachedFromTags(inference.tags);
        const statusCode = statusCodeFromTags(inference.tags);
        const tokens = totalTokens(inference);
        const outputTps = outputTpsExcludingTtft({
          outputTokens: inference.output_tokens,
          durationMs: inference.processing_time_ms,
          ttftMs: inference.ttft_ms,
        });
        const { userTags } = splitInferenceTags(inference.tags);
        const tagEntries = Object.entries(userTags).slice(0, 3);
        const inferenceUrl = toInferenceUrl(inference.id);

        return (
          <TableRow
            key={inference.id}
            id={inference.id}
            className="cursor-pointer"
            onClick={() => navigate(inferenceUrl)}
          >
            <TableCell className="text-sm">
              <span className="inline-flex items-center gap-1">
                <TableItemTime
                  timestamp={uuidv7ToTimestamp(inference.id).toISOString()}
                />
                {data.protection[inference.id] !== undefined && (
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <ShieldCheck
                        className="text-muted-foreground h-4 w-4"
                        aria-label="Protected from cleanup"
                      />
                    </TooltipTrigger>
                    <TooltipContent>
                      Protected from cleanup — archived permanently
                    </TooltipContent>
                  </Tooltip>
                )}
              </span>
            </TableCell>
            <TableCell>
              {provider ? (
                <Badge variant="secondary">{provider}</Badge>
              ) : (
                <span className="text-sm text-muted-foreground">—</span>
              )}
            </TableCell>
            <TableCell className="font-mono text-sm">
              <span className="align-middle">{inference.variant_name}</span>
              {fallbackCount > 0 && (
                <Badge
                  variant="outline"
                  className="ml-2 text-[10px] border-yellow-500/40 text-yellow-700 dark:text-yellow-300"
                >
                  🔀 {fallbackCount}
                </Badge>
              )}
            </TableCell>
            <TableCell className="text-right text-sm">
              {tokens === undefined ? (
                <span className="text-muted-foreground">—</span>
              ) : (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span>{tokens.toLocaleString()}</span>
                  </TooltipTrigger>
                  <TooltipContent>
                    {(inference.input_tokens ?? 0).toLocaleString()} in /{" "}
                    {(inference.output_tokens ?? 0).toLocaleString()} out
                  </TooltipContent>
                </Tooltip>
              )}
            </TableCell>
            <TableCell className="text-right text-sm">
              {inference.cost == null ? (
                <span className="text-muted-foreground">—</span>
              ) : (
                formatCost(inference.cost, inference.currency ?? undefined)
              )}
            </TableCell>
            <TableCell className="text-right text-sm">
              {inference.processing_time_ms == null ? (
                <span className="text-muted-foreground">—</span>
              ) : (
                `${inference.processing_time_ms}ms`
              )}
            </TableCell>
            <TableCell className="text-right text-sm">
              {toFiniteMs(inference.ttft_ms) == null ? (
                <span className="text-muted-foreground">—</span>
              ) : (
                `${inference.ttft_ms}ms`
              )}
            </TableCell>
            <TableCell className="text-right text-sm">
              {outputTps == null ? (
                <span className="text-muted-foreground">—</span>
              ) : (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span>{formatOutputTps(outputTps)}</span>
                  </TooltipTrigger>
                  <TooltipContent>
                    Output tokens per second, excluding TTFT
                  </TooltipContent>
                </Tooltip>
              )}
            </TableCell>
            <TableCell>
              {cached ? (
                <Badge variant="outline">cached</Badge>
              ) : (
                <span className="text-sm text-muted-foreground">—</span>
              )}
            </TableCell>
            <TableCell>
              <Badge
                variant={statusCode >= 400 ? "destructive" : "secondary"}
                className={
                  statusCode >= 200 && statusCode < 300
                    ? "border-transparent bg-emerald-600 text-white"
                    : undefined
                }
              >
                {statusCode}
              </Badge>
            </TableCell>
            <TableCell>
              {tagEntries.length === 0 ? (
                <span className="text-sm text-muted-foreground">—</span>
              ) : (
                <div className="flex max-w-[180px] flex-wrap gap-1">
                  {tagEntries.map(([key, value]) => (
                    <Badge
                      key={key}
                      variant="outline"
                      className="max-w-full truncate text-[10px]"
                      title={`${key}=${value}`}
                    >
                      {key}={value}
                    </Badge>
                  ))}
                </div>
              )}
            </TableCell>
            <TableCell className="w-[44px] px-1">
              <FunctionIcon
                functionName={inference.function_name}
                functionType={kind}
                snapshotHash={inference.snapshot_hash}
              />
            </TableCell>
            <TableCell className="w-[36px] px-1">
              <JumpIcon
                to={inferenceUrl}
                label={`Open inference ${inference.id}`}
              >
                <Eye className="h-4 w-4" />
              </JumpIcon>
            </TableCell>
            <TableCell className="w-[36px] px-1">
              <JumpIcon
                to={toEpisodeUrl(inference.episode_id)}
                label={`Open episode ${inference.episode_id}`}
              >
                <Layers className="h-4 w-4" />
              </JumpIcon>
            </TableCell>
          </TableRow>
        );
      })}
    </>
  );
}

function FunctionIcon({
  functionName,
  functionType,
  snapshotHash,
}: {
  functionName: string;
  functionType: string;
  snapshotHash?: string;
}) {
  const icon = getFunctionTypeIcon(functionType);
  const standalone = isStandaloneFunctionName(functionName);
  const className = `${icon.iconBg} inline-flex rounded-sm p-0.5`;
  const content = (
    <span className={className} aria-label={`Function ${functionName}`}>
      {icon.icon}
    </span>
  );

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        {standalone ? (
          <span
            className="inline-flex"
            onClick={(event) => event.stopPropagation()}
          >
            {content}
          </span>
        ) : (
          <Link
            to={toFunctionUrl(functionName, snapshotHash)}
            className="inline-flex"
            onClick={(event) => event.stopPropagation()}
          >
            {content}
          </Link>
        )}
      </TooltipTrigger>
      <TooltipContent>
        <div className="font-mono text-xs">{functionName}</div>
        {icon.label && (
          <div className="text-[10px] text-white/70">{icon.label}</div>
        )}
      </TooltipContent>
    </Tooltip>
  );
}

function JumpIcon({
  to,
  label,
  children,
}: {
  to: string;
  label: string;
  children: ReactNode;
}) {
  return (
    <Button variant="ghost" size="iconSm" className="h-7 w-7" asChild>
      <Link
        to={to}
        aria-label={label}
        onClick={(event) => event.stopPropagation()}
      >
        {children}
      </Link>
    </Button>
  );
}

function totalTokens(inference: InferenceListRow): number | undefined {
  if (inference.input_tokens == null && inference.output_tokens == null) {
    return undefined;
  }
  return (inference.input_tokens ?? 0) + (inference.output_tokens ?? 0);
}

function PaginationButtons({
  data,
  limit,
}: {
  data: InferencesData;
  limit: number;
}) {
  const { inferences, hasNextPage, hasPreviousPage } = data;
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  const topInference = inferences.at(0);
  const bottomInference = inferences.at(inferences.length - 1);

  const buildSearchParams = () => {
    const params = new URLSearchParams(searchParams);
    params.set("limit", String(limit));
    params.delete("before");
    params.delete("after");
    return params;
  };

  const handleNextPage = () => {
    if (bottomInference) {
      const params = buildSearchParams();
      params.set("before", bottomInference.id);
      navigate(`?${params.toString()}`, {
        preventScrollReset: true,
      });
    }
  };

  const handlePreviousPage = () => {
    if (topInference) {
      const params = buildSearchParams();
      params.set("after", topInference.id);
      navigate(`?${params.toString()}`, {
        preventScrollReset: true,
      });
    }
  };

  return (
    <PageButtons
      onPreviousPage={handlePreviousPage}
      onNextPage={handleNextPage}
      disablePrevious={!hasPreviousPage}
      disableNext={!hasNextPage}
    />
  );
}

export default function InferencesTable({
  data,
  limit,
}: {
  data: Promise<InferencesData>;
  limit: number;
}) {
  const location = useLocation();

  return (
    <div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Time</TableHead>
            <TableHead>Provider</TableHead>
            <TableHead>Model</TableHead>
            <TableHead className="text-right">Tokens</TableHead>
            <TableHead className="text-right">Cost</TableHead>
            <TableHead className="text-right">Latency</TableHead>
            <TableHead
              className="text-right"
              title="Time to first token (streaming only)"
            >
              TTFT
            </TableHead>
            <TableHead
              className="text-right"
              title="Output tokens per second, excluding TTFT"
            >
              Output tok/s
            </TableHead>
            <TableHead>Cache</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Tags</TableHead>
            <TableHead className="w-[44px]">
              <span className="sr-only">Function</span>
            </TableHead>
            <TableHead className="w-[36px]">
              <span className="sr-only">Inference</span>
            </TableHead>
            <TableHead className="w-[36px]">
              <span className="sr-only">Episode</span>
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          <Suspense key={location.key} fallback={<SkeletonRows />}>
            <Await
              resolve={data}
              errorElement={
                <TableAsyncErrorState
                  colSpan={COLUMN_COUNT}
                  defaultMessage="Failed to load inferences"
                />
              }
            >
              {(resolvedData) => <TableRows data={resolvedData} />}
            </Await>
          </Suspense>
        </TableBody>
      </Table>

      <Suspense key={location.key} fallback={<PageButtons disabled />}>
        <Await resolve={data} errorElement={<PageButtons disabled />}>
          {(resolvedData) => (
            <PaginationButtons data={resolvedData} limit={limit} />
          )}
        </Await>
      </Suspense>
    </div>
  );
}

// Modified by Delta-AI under Apache 2.0
import { Await, useNavigate } from "react-router";
import { Suspense } from "react";
import type { RouteHandle } from "react-router";
import type { Route } from "./+types/route";
import {
  Activity,
  CheckCircle2,
  CircleDollarSign,
  Cpu,
  Database,
  HardDrive,
  Zap,
} from "lucide-react";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import { Button } from "~/components/ui/button";
import { Card, CardContent } from "~/components/ui/card";
import { Checkbox } from "~/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";
import { StatCard } from "~/components/analysis/StatCard";
import {
  EmbeddingModelBars,
  LatencyChart,
  ModelTable,
  OutputTpsChart,
  ProviderChart,
  RequestsChart,
  TokenUsageChart,
  TtftChart,
} from "~/components/analysis/charts";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import {
  getPostgresClient,
  isPostgresAvailable,
} from "~/utils/postgres.server";
import { logger } from "~/utils/logger";
import { useConfig } from "~/context/config";
import {
  ANALYSIS_RANGES,
  analysisModelsForKind,
  analysisSearchParams,
  formatCompactCount,
  formatInputCacheHitDescription,
  parseAnalysisQuery,
  rangeDescription,
  type AnalysisKind,
  type AnalysisQueryValues,
  type AnalysisRange,
  type AnalysisResponse,
} from "./analysisQuery";
import {
  apiKeysForSelect,
  formatApiKeyOption,
} from "~/routes/observability/inferences/inferenceQuery";
import type { KeyInfo } from "~/types/tensorzero";
import { formatCost } from "~/utils/cost";

export const handle: RouteHandle = {
  crumb: () => ["Analysis"],
};

export async function loader({ request }: Route.LoaderArgs) {
  const query = parseAnalysisQuery(new URL(request.url).searchParams);
  const client = getTensorZeroClient();
  const analysisPromise = client
    .getSynapseAnalysis({
      range: query.range,
      kind: query.kind,
      apiKey: query.apiKey,
      model: query.model,
      cacheMissOnly: query.cacheMissOnly,
    })
    .catch((error) => {
      logger.error(error);
      return {
        error:
          error instanceof Error ? error.message : "Failed to load analysis",
      };
    });
  const apiKeysPromise = (async (): Promise<KeyInfo[]> => {
    if (!isPostgresAvailable()) {
      return [];
    }
    try {
      const postgres = await getPostgresClient();
      return await postgres.listApiKeys(1000, 0);
    } catch (error) {
      logger.error("Failed to list API keys for analysis filters", error);
      return [];
    }
  })();
  return {
    query,
    analysisPromise,
    apiKeys: await apiKeysPromise,
  };
}

export default function AnalysisPage({ loaderData }: Route.ComponentProps) {
  const { query, analysisPromise, apiKeys } = loaderData;
  return (
    <PageLayout>
      <PageHeader heading="Analysis" />
      <SectionLayout>
        <AnalysisFilters query={query} apiKeys={apiKeys} />
        <Suspense
          fallback={<p className="text-muted-foreground">Loading analysis…</p>}
        >
          <Await resolve={analysisPromise}>
            {(result) =>
              "error" in result ? (
                <Card>
                  <CardContent className="flex flex-col items-center justify-center py-12">
                    <p className="text-destructive">{result.error}</p>
                    <p className="text-muted-foreground mt-1 text-sm">
                      Analysis needs Postgres observability and a running
                      gateway.
                    </p>
                  </CardContent>
                </Card>
              ) : (
                <AnalysisBody query={query} data={result} />
              )
            }
          </Await>
        </Suspense>
      </SectionLayout>
    </PageLayout>
  );
}

function AnalysisFilters({
  query,
  apiKeys,
}: {
  query: AnalysisQueryValues;
  apiKeys: KeyInfo[];
}) {
  const navigate = useNavigate();
  const config = useConfig();
  const models = analysisModelsForKind(query.kind, config);
  const keyOptions = apiKeysForSelect(apiKeys, query.apiKey);
  const go = (next: AnalysisQueryValues) => {
    const params = analysisSearchParams(next);
    const qs = params.toString();
    navigate(qs ? `?${qs}` : ".", { preventScrollReset: true });
  };

  return (
    <div className="mb-6 space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        {ANALYSIS_RANGES.map((range) => (
          <Button
            key={range}
            type="button"
            size="sm"
            variant={query.range === range ? "default" : "outline"}
            onClick={() => go({ ...query, range })}
          >
            {range}
          </Button>
        ))}
      </div>
      <Card>
        <CardContent className="flex flex-col gap-4 pt-6 lg:flex-row lg:items-end lg:justify-between">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-end">
            <label className="space-y-2 text-sm font-medium">
              API Key
              <Select
                value={query.apiKey || "all"}
                onValueChange={(value) =>
                  go({ ...query, apiKey: value === "all" ? "" : value })
                }
              >
                <SelectTrigger className="w-full min-w-[240px] lg:w-[280px]">
                  <SelectValue placeholder="All API Keys" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All API Keys</SelectItem>
                  {keyOptions.map((key) => (
                    <SelectItem key={key.public_id} value={key.public_id}>
                      {formatApiKeyOption(key)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </label>
            <label className="space-y-2 text-sm font-medium">
              Model
              <Select
                value={query.model || "all"}
                onValueChange={(value) =>
                  go({ ...query, model: value === "all" ? "" : value })
                }
              >
                <SelectTrigger className="w-full min-w-[240px] lg:w-[280px]">
                  <SelectValue placeholder="All Models" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Models</SelectItem>
                  {models.map((model) => (
                    <SelectItem key={model} value={model}>
                      {model}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-muted-foreground text-xs">
                {query.kind === "chat"
                  ? "Applies to Chat Analysis only."
                  : "Applies to Embedding Analysis only."}
              </p>
            </label>
          </div>
          <div className="bg-muted/50 inline-flex self-start rounded-lg border p-1 lg:self-auto">
            <KindButton
              current={query.kind}
              value="chat"
              label="Chat Analysis"
              onSelect={(kind) => go({ ...query, kind, model: "" })}
            />
            <KindButton
              current={query.kind}
              value="embedding"
              label="Embedding Analysis"
              onSelect={(kind) => go({ ...query, kind, model: "" })}
            />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function KindButton({
  current,
  value,
  label,
  onSelect,
}: {
  current: AnalysisKind;
  value: AnalysisKind;
  label: string;
  onSelect: (kind: AnalysisKind) => void;
}) {
  return (
    <Button
      type="button"
      size="sm"
      variant={current === value ? "default" : "ghost"}
      onClick={() => onSelect(value)}
    >
      {label}
    </Button>
  );
}

function AnalysisBody({
  query,
  data,
}: {
  query: AnalysisQueryValues;
  data: AnalysisResponse;
}) {
  const range: AnalysisRange = query.range;
  const cacheMiss = query.cacheMissOnly;
  const costs = Object.entries(data.total_cost_by_currency).sort(
    ([left], [right]) => left.localeCompare(right),
  );
  const navigate = useNavigate();
  const toggleCacheMiss = () => {
    const params = analysisSearchParams({
      ...query,
      cacheMissOnly: !query.cacheMissOnly,
    });
    const qs = params.toString();
    navigate(qs ? `?${qs}` : ".", { preventScrollReset: true });
  };

  return (
    <section className="space-y-4">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h2 className="text-lg font-semibold">
            {query.kind === "embedding"
              ? "Embedding Analysis"
              : "Chat Analysis"}
          </h2>
          <p className="text-muted-foreground text-sm">
            {query.kind === "embedding"
              ? "Successful embedding requests, cache usage, provider mix, and latency."
              : "Successful chat requests, latency, TTFT, output speed, cache, and tokens."}
          </p>
        </div>
        <label className="flex items-center gap-2 rounded-lg border px-3 py-2 text-sm font-medium">
          <Checkbox
            checked={cacheMiss}
            onCheckedChange={() => toggleCacheMiss()}
            aria-label="Only cache misses"
          />
          Only cache misses
        </label>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4 xl:grid-cols-4">
        <StatCard
          title="Total Requests"
          value={formatCompactCount(data.total_requests)}
          icon={Activity}
          description={rangeDescription(range)}
        />
        <StatCard
          title="Success Rate"
          value={`${data.success_rate.toFixed(1)}%`}
          icon={CheckCircle2}
          description={`${formatCompactCount(data.total_requests)} / ${formatCompactCount(data.total_responses)} successful`}
        />
        {cacheMiss ? (
          <StatCard
            title="Cache Scope"
            value="Misses only"
            icon={Zap}
            description="Cached responses excluded"
          />
        ) : (
          <StatCard
            title="Cache Hit Rate"
            value={`${data.cache_hit_rate.toFixed(1)}%`}
            icon={Zap}
            description="Cached responses"
          />
        )}
        <StatCard
          title="Input Cache Hit Rate"
          value={`${data.input_cache_hit_rate.toFixed(1)}%`}
          icon={HardDrive}
          description={formatInputCacheHitDescription(
            data.cache_read_input_tokens,
            data.total_input_tokens,
          )}
        />
        <StatCard
          title="Avg Latency"
          value={
            data.avg_latency != null
              ? `${Math.round(data.avg_latency)}ms`
              : "N/A"
          }
          icon={Cpu}
          description="Response time"
        />
        <StatCard
          title="Total Tokens"
          value={formatCompactCount(data.total_tokens)}
          icon={Database}
          description={
            query.kind === "embedding"
              ? "Token consumption"
              : `${formatCompactCount(data.total_input_tokens)} in / ${formatCompactCount(data.total_output_tokens)} out`
          }
        />
        {costs.length === 0 ? (
          <StatCard
            title="Cost"
            value="N/A"
            icon={CircleDollarSign}
            description="No billed usage"
          />
        ) : (
          costs.map(([currency, amount]) => (
            <StatCard
              key={currency}
              title={`Cost (${currency})`}
              value={formatCost(amount, currency)}
              icon={CircleDollarSign}
              description={`Billed in ${currency}`}
            />
          ))
        )}
      </div>

      {query.kind === "embedding" ? (
        <div className="grid gap-4 md:grid-cols-2">
          <TokenUsageChart data={data.token_usage_over_time} embedding />
          <LatencyChart
            data={data.latency_over_time}
            modelStats={data.model_latency_stats}
          />
          <ProviderChart data={data.provider_stats} />
          <EmbeddingModelBars data={data.model_stats} />
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2">
          <RequestsChart data={data.requests_over_time} />
          <LatencyChart
            data={data.latency_over_time}
            modelStats={data.model_latency_stats}
          />
          <TtftChart data={data.ttft_over_time} />
          <OutputTpsChart data={data.output_tps_over_time} />
          <ProviderChart data={data.provider_stats} />
          <ModelTable data={data.model_stats} />
          <TokenUsageChart data={data.token_usage_over_time} />
        </div>
      )}
    </section>
  );
}

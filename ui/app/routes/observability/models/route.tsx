// Modified by Delta-AI under Apache 2.0
import { data, Await, Form } from "react-router";
import { Suspense } from "react";
import type { ActionFunctionArgs, RouteHandle } from "react-router";
import type { Route } from "./+types/route";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import type { TimeWindow } from "~/types/tensorzero";
import { ModelUsage } from "~/components/model/ModelUsage";
import { ModelLatency } from "~/components/model/ModelLatency";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
  SectionsGroup,
  SectionHeader,
} from "~/components/layout/PageLayout";
import { logger } from "~/utils/logger";
import type {
  SynapseAnalyticsRow,
  SynapseBalances,
} from "~/utils/tensorzero/tensorzero";
import { Button } from "~/components/ui/button";

export const handle: RouteHandle = {
  crumb: () => ["Models"],
};

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const usageTimeGranularityParam =
    url.searchParams.get("usageTimeGranularity") || "week";
  const latencyTimeGranularityParam =
    url.searchParams.get("latencyTimeGranularity") || "week";

  // Validate TimeWindow type
  const validTimeWindows: TimeWindow[] = [
    "hour",
    "day",
    "week",
    "month",
    "cumulative",
  ];
  if (!validTimeWindows.includes(usageTimeGranularityParam as TimeWindow)) {
    throw data(
      `Invalid usage time granularity: ${usageTimeGranularityParam}. Must be one of: ${validTimeWindows.join(", ")}`,
      { status: 400 },
    );
  }
  const usageTimeGranularity = usageTimeGranularityParam as TimeWindow;
  if (!validTimeWindows.includes(latencyTimeGranularityParam as TimeWindow)) {
    throw data(
      `Invalid latency time granularity: ${latencyTimeGranularityParam}. Must be one of: ${validTimeWindows.join(", ")}`,
      { status: 400 },
    );
  }
  const latencyTimeGranularity = latencyTimeGranularityParam as TimeWindow;

  const numPeriods = parseInt(url.searchParams.get("usageNumPeriods") || "10");
  const client = getTensorZeroClient();
  const modelUsageTimeseriesPromise = client
    .getModelUsageTimeseries(usageTimeGranularity, numPeriods)
    .then((response) => response.data);
  const modelLatencyQuantilesPromise = client.getModelLatencyQuantiles(
    latencyTimeGranularity,
  );
  const synapseAnalyticsPromise = (async () => {
    const to = new Date();
    const from = new Date(to.getTime() - 7 * 24 * 60 * 60 * 1000);
    try {
      const response = await client.getSynapseAnalytics(
        from.toISOString(),
        to.toISOString(),
      );
      return response.data;
    } catch (error) {
      logger.error(error);
      return [] as SynapseAnalyticsRow[];
    }
  })();
  const synapseBalancesPromise = client.getSynapseBalances().catch((error) => {
    logger.error(error);
    return { deepseek: null, openrouter: null } as SynapseBalances;
  });
  return {
    modelUsageTimeseriesPromise,
    usageTimeGranularity,
    latencyTimeGranularity,
    modelLatencyQuantilesPromise,
    synapseAnalyticsPromise,
    synapseBalancesPromise,
  };
}

export async function action({ request }: ActionFunctionArgs) {
  const form = await request.formData();
  if (form.get("intent") !== "usage_csv") {
    throw data("Unknown action", { status: 400 });
  }
  const to = new Date();
  const from = new Date(to.getTime() - 7 * 24 * 60 * 60 * 1000);
  try {
    const csv = await getTensorZeroClient().getSynapseUsageExport(
      from.toISOString(),
      to.toISOString(),
    );
    return new Response(csv, {
      headers: {
        "Content-Type": "text/csv; charset=utf-8",
        "Content-Disposition":
          'attachment; filename="synapse-deepseek-usage.csv"',
      },
    });
  } catch (error) {
    logger.error(error);
    throw data(error instanceof Error ? error.message : String(error), {
      status: 502,
    });
  }
}

export default function ModelsPage({ loaderData }: Route.ComponentProps) {
  const {
    modelUsageTimeseriesPromise,
    modelLatencyQuantilesPromise,
    synapseAnalyticsPromise,
    synapseBalancesPromise,
  } = loaderData;

  return (
    <PageLayout>
      <PageHeader heading="Models" />

      <SectionsGroup>
        <SectionLayout>
          <SectionHeader heading="Usage" />
          <ModelUsage modelUsageDataPromise={modelUsageTimeseriesPromise} />
        </SectionLayout>
        <SectionLayout>
          <SectionHeader heading="Latency" />
          <ModelLatency
            modelLatencyResponsePromise={modelLatencyQuantilesPromise}
          />
        </SectionLayout>
        <SectionLayout>
          <SectionHeader heading="Provider balances" />
          <Suspense fallback={<p>Loading balances…</p>}>
            <Await resolve={synapseBalancesPromise}>
              {(balances: SynapseBalances) => (
                <div className="grid gap-4 md:grid-cols-2">
                  <pre className="bg-bg-hover overflow-auto rounded-md p-4 text-sm">
                    DeepSeek{"\n"}
                    {balances.deepseek
                      ? JSON.stringify(balances.deepseek, null, 2)
                      : "No DEEPSEEK_API_KEY"}
                  </pre>
                  <pre className="bg-bg-hover overflow-auto rounded-md p-4 text-sm">
                    OpenRouter{"\n"}
                    {balances.openrouter
                      ? JSON.stringify(balances.openrouter, null, 2)
                      : "No OPENROUTER_API_KEY"}
                  </pre>
                </div>
              )}
            </Await>
          </Suspense>
        </SectionLayout>
        <SectionLayout>
          <SectionHeader heading="Request mix (last 7 days)" />
          <Form method="post" reloadDocument className="mb-3">
            <input type="hidden" name="intent" value="usage_csv" />
            <Button type="submit">Download DeepSeek CNY CSV</Button>
          </Form>
          <Suspense fallback={<p>Loading analytics…</p>}>
            <Await resolve={synapseAnalyticsPromise}>
              {(rows: SynapseAnalyticsRow[]) =>
                rows.length === 0 ? (
                  <p className="text-sm">
                    No Postgres analytics yet. Point observability at Postgres
                    and send traffic.
                  </p>
                ) : (
                  <div className="overflow-auto">
                    <table className="w-full text-left text-sm">
                      <thead>
                        <tr>
                          <th className="p-2">Model</th>
                          <th className="p-2">Kind</th>
                          <th className="p-2">Requests</th>
                          <th className="p-2">Input tokens</th>
                          <th className="p-2">Output tokens</th>
                          <th className="p-2">Avg latency ms</th>
                          <th className="p-2">Avg TTFT ms</th>
                          <th className="p-2">Output TPS (ex-TTFT)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {rows.map((row) => (
                          <tr key={`${row.kind}:${row.model_name}`}>
                            <td className="p-2">{row.model_name}</td>
                            <td className="p-2">{row.kind}</td>
                            <td className="p-2">{row.requests}</td>
                            <td className="p-2">{row.input_tokens}</td>
                            <td className="p-2">{row.output_tokens}</td>
                            <td className="p-2">
                              {row.avg_latency_ms?.toFixed(1) ?? "—"}
                            </td>
                            <td className="p-2">
                              {row.avg_ttft_ms?.toFixed(1) ?? "—"}
                            </td>
                            <td className="p-2">
                              {row.output_tps_excluding_ttft?.toFixed(2) ?? "—"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )
              }
            </Await>
          </Suspense>
        </SectionLayout>
      </SectionsGroup>
    </PageLayout>
  );
}

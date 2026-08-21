// Modified by Delta-AI under Apache 2.0
import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  XAxis,
  YAxis,
} from "recharts";
import { Badge } from "~/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { ChartContainer, ChartTooltip } from "~/components/ui/chart";
import { CHART_COLORS } from "~/utils/chart";
import {
  formatBucketLabel,
  type AnalysisCountPoint,
  type AnalysisModelStats,
  type AnalysisPercentilePoint,
  type AnalysisProviderStats,
  type AnalysisTokenPoint,
} from "~/routes/observability/analysis/analysisQuery";

const LINE_COLORS = {
  p50: CHART_COLORS[1],
  p90: CHART_COLORS[0],
  p99: CHART_COLORS[2],
  avg: CHART_COLORS[3],
  input: CHART_COLORS[0],
  output: CHART_COLORS[1],
  count: CHART_COLORS[0],
} as const;

function EmptyChart({ title }: { title: string }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent className="text-muted-foreground flex h-[300px] items-center justify-center text-sm">
        No data available
      </CardContent>
    </Card>
  );
}

function PercentileToggles<T extends string>({
  options,
  selected,
  onToggle,
}: {
  options: Array<{ value: T; label: string }>;
  selected: T[];
  onToggle: (value: T) => void;
}) {
  return (
    <div className="mt-2 flex gap-1">
      {options.map((option) => {
        const active = selected.includes(option.value);
        return (
          <button
            key={option.value}
            type="button"
            onClick={() => onToggle(option.value)}
            className={`rounded-md px-2 py-1 text-xs transition-colors ${
              active
                ? "bg-primary text-primary-foreground"
                : "bg-muted text-muted-foreground hover:bg-muted/80"
            }`}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

export function RequestsChart({ data }: { data: AnalysisCountPoint[] }) {
  if (data.length === 0) {
    return <EmptyChart title="Requests Over Time" />;
  }
  const chartData = data.map((point) => ({
    date: formatBucketLabel(point.date),
    count: point.count,
  }));
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Requests Over Time</CardTitle>
      </CardHeader>
      <CardContent>
        <ChartContainer
          className="h-[300px]"
          config={{ count: { label: "Requests", color: LINE_COLORS.count } }}
        >
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 12 }} />
            <ChartTooltip />
            <Line
              type="monotone"
              dataKey="count"
              name="Requests"
              stroke="var(--color-count)"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}

type LatencyKey = "p50" | "p90" | "p99" | "avg";

export function LatencyChart({
  data,
  modelStats,
}: {
  data: AnalysisPercentilePoint[];
  modelStats: AnalysisModelStats[];
}) {
  const [selected, setSelected] = useState<LatencyKey[]>(["p50", "p90", "p99"]);
  if (data.length === 0) {
    return <EmptyChart title="Latency Over Time" />;
  }
  const chartData = data.map((point) => ({
    date: formatBucketLabel(point.date),
    p50: point.p50 == null ? null : Math.round(point.p50),
    p90: point.p90 == null ? null : Math.round(point.p90),
    p99: point.p99 == null ? null : Math.round(point.p99),
    avg: point.avg == null ? null : Math.round(point.avg),
  }));
  const toggle = (key: LatencyKey) => {
    setSelected((current) => {
      if (current.includes(key)) {
        return current.length > 1
          ? current.filter((item) => item !== key)
          : current;
      }
      return [...current, key];
    });
  };
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Latency Over Time</CardTitle>
        <PercentileToggles
          options={[
            { value: "p50", label: "P50" },
            { value: "p90", label: "P90" },
            { value: "p99", label: "P99" },
            { value: "avg", label: "Average" },
          ]}
          selected={selected}
          onToggle={toggle}
        />
      </CardHeader>
      <CardContent>
        {modelStats.length > 0 ? (
          <p className="text-muted-foreground mb-3 text-xs">
            {modelStats.length} models in range
          </p>
        ) : null}
        <ChartContainer
          className="h-[300px]"
          config={{
            p50: { label: "P50", color: LINE_COLORS.p50 },
            p90: { label: "P90", color: LINE_COLORS.p90 },
            p99: { label: "P99", color: LINE_COLORS.p99 },
            avg: { label: "Average", color: LINE_COLORS.avg },
          }}
        >
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 12 }} unit="ms" />
            <ChartTooltip />
            <Legend />
            {selected.includes("p50") ? (
              <Line
                type="monotone"
                dataKey="p50"
                name="P50"
                stroke="var(--color-p50)"
                strokeWidth={2}
                dot={false}
                connectNulls
              />
            ) : null}
            {selected.includes("p90") ? (
              <Line
                type="monotone"
                dataKey="p90"
                name="P90"
                stroke="var(--color-p90)"
                strokeWidth={2}
                dot={false}
                connectNulls
              />
            ) : null}
            {selected.includes("p99") ? (
              <Line
                type="monotone"
                dataKey="p99"
                name="P99"
                stroke="var(--color-p99)"
                strokeWidth={2}
                dot={false}
                connectNulls
              />
            ) : null}
            {selected.includes("avg") ? (
              <Line
                type="monotone"
                dataKey="avg"
                name="Average"
                stroke="var(--color-avg)"
                strokeWidth={2}
                dot={false}
                connectNulls
              />
            ) : null}
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}

type TpsKey = "p50" | "p90" | "avg";

export function TtftChart({ data }: { data: AnalysisPercentilePoint[] }) {
  const [selected, setSelected] = useState<TpsKey[]>(["p50", "p90", "avg"]);
  if (data.length === 0) {
    return <EmptyChart title="TTFT" />;
  }
  const chartData = data.map((point) => ({
    date: formatBucketLabel(point.date),
    p50: point.p50 == null ? null : Math.round(point.p50),
    p90: point.p90 == null ? null : Math.round(point.p90),
    avg: point.avg == null ? null : Math.round(point.avg),
  }));
  const toggle = (key: TpsKey) => {
    setSelected((current) => {
      if (current.includes(key)) {
        return current.length > 1
          ? current.filter((item) => item !== key)
          : current;
      }
      return [...current, key];
    });
  };
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">TTFT</CardTitle>
        <PercentileToggles
          options={[
            { value: "p50", label: "P50" },
            { value: "p90", label: "P90" },
            { value: "avg", label: "Average" },
          ]}
          selected={selected}
          onToggle={toggle}
        />
      </CardHeader>
      <CardContent>
        <ChartContainer
          className="h-[300px]"
          config={{
            p50: { label: "P50", color: LINE_COLORS.p50 },
            p90: { label: "P90", color: LINE_COLORS.p90 },
            avg: { label: "Average", color: LINE_COLORS.avg },
          }}
        >
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 12 }} unit="ms" />
            <ChartTooltip />
            <Legend />
            {selected.includes("p50") ? (
              <Line
                type="monotone"
                dataKey="p50"
                name="P50"
                stroke="var(--color-p50)"
                strokeWidth={2}
                dot={false}
                connectNulls
              />
            ) : null}
            {selected.includes("p90") ? (
              <Line
                type="monotone"
                dataKey="p90"
                name="P90"
                stroke="var(--color-p90)"
                strokeWidth={2}
                dot={false}
                connectNulls
              />
            ) : null}
            {selected.includes("avg") ? (
              <Line
                type="monotone"
                dataKey="avg"
                name="Average"
                stroke="var(--color-avg)"
                strokeWidth={2}
                dot={false}
                connectNulls
              />
            ) : null}
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}

export function OutputTpsChart({ data }: { data: AnalysisPercentilePoint[] }) {
  const [selected, setSelected] = useState<TpsKey[]>(["p50", "p90", "avg"]);
  if (data.length === 0) {
    return <EmptyChart title="Output tokens/sec" />;
  }
  const chartData = data.map((point) => ({
    date: formatBucketLabel(point.date),
    p50: point.p50 == null ? null : Math.round(point.p50 * 10) / 10,
    p90: point.p90 == null ? null : Math.round(point.p90 * 10) / 10,
    avg: point.avg == null ? null : Math.round(point.avg * 10) / 10,
  }));
  const toggle = (key: TpsKey) => {
    setSelected((current) => {
      if (current.includes(key)) {
        return current.length > 1
          ? current.filter((item) => item !== key)
          : current;
      }
      return [...current, key];
    });
  };
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Output tokens/sec</CardTitle>
        <PercentileToggles
          options={[
            { value: "p50", label: "P50" },
            { value: "p90", label: "P90" },
            { value: "avg", label: "Average" },
          ]}
          selected={selected}
          onToggle={toggle}
        />
      </CardHeader>
      <CardContent>
        <ChartContainer
          className="h-[300px]"
          config={{
            p50: { label: "P50", color: LINE_COLORS.p50 },
            p90: { label: "P90", color: LINE_COLORS.p90 },
            avg: { label: "Average", color: LINE_COLORS.avg },
          }}
        >
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 12 }} unit=" tok/s" />
            <ChartTooltip />
            <Legend />
            {selected.includes("p50") ? (
              <Line
                type="monotone"
                dataKey="p50"
                name="P50"
                stroke="var(--color-p50)"
                strokeWidth={2}
                dot={false}
                connectNulls
              />
            ) : null}
            {selected.includes("p90") ? (
              <Line
                type="monotone"
                dataKey="p90"
                name="P90"
                stroke="var(--color-p90)"
                strokeWidth={2}
                dot={false}
                connectNulls
              />
            ) : null}
            {selected.includes("avg") ? (
              <Line
                type="monotone"
                dataKey="avg"
                name="Average"
                stroke="var(--color-avg)"
                strokeWidth={2}
                dot={false}
                connectNulls
              />
            ) : null}
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}

export function ProviderChart({ data }: { data: AnalysisProviderStats[] }) {
  if (data.length === 0) {
    return <EmptyChart title="Requests by Provider" />;
  }
  const chartData = data.map((row, index) => ({
    name: row.provider,
    value: row.count,
    fill: CHART_COLORS[index % CHART_COLORS.length],
  }));
  const config = Object.fromEntries(
    chartData.map((row) => [row.name, { label: row.name, color: row.fill }]),
  );
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Requests by Provider</CardTitle>
      </CardHeader>
      <CardContent>
        <ChartContainer className="h-[300px]" config={config}>
          <PieChart>
            <Pie
              data={chartData}
              dataKey="value"
              nameKey="name"
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={100}
              paddingAngle={2}
              label={({ name, percent }: { name?: string; percent?: number }) =>
                `${name ?? ""} (${((percent ?? 0) * 100).toFixed(0)}%)`
              }
            >
              {chartData.map((row) => (
                <Cell key={row.name} fill={row.fill} />
              ))}
            </Pie>
            <ChartTooltip />
            <Legend />
          </PieChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}

export function ModelTable({ data }: { data: AnalysisModelStats[] }) {
  if (data.length === 0) {
    return <EmptyChart title="Requests by Model" />;
  }
  const sorted = [...data].sort((a, b) => b.count - a.count);
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Requests by Model</CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <div className="max-h-[300px] overflow-auto">
          <table className="w-full">
            <thead className="bg-background sticky top-0">
              <tr className="bg-muted/50 border-b">
                <th className="text-muted-foreground px-4 py-2 text-left text-xs font-medium">
                  Model
                </th>
                <th className="text-muted-foreground px-4 py-2 text-left text-xs font-medium">
                  Provider
                </th>
                <th className="text-muted-foreground px-4 py-2 text-right text-xs font-medium">
                  Requests
                </th>
                <th className="text-muted-foreground px-4 py-2 text-right text-xs font-medium">
                  Avg Latency
                </th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((item) => (
                <tr
                  key={`${item.provider}-${item.model}`}
                  className="border-b last:border-0"
                >
                  <td className="px-4 py-2 font-mono text-sm">
                    {item.model.length > 30
                      ? `${item.model.slice(0, 27)}...`
                      : item.model}
                  </td>
                  <td className="px-4 py-2">
                    <Badge variant="outline" className="text-xs">
                      {item.provider}
                    </Badge>
                  </td>
                  <td className="px-4 py-2 text-right text-sm font-medium">
                    {item.count.toLocaleString()}
                  </td>
                  <td className="text-muted-foreground px-4 py-2 text-right text-sm">
                    {item.avg_latency != null
                      ? `${Math.round(item.avg_latency)}ms`
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}

export function TokenUsageChart({
  data,
  embedding,
}: {
  data: AnalysisTokenPoint[];
  embedding?: boolean;
}) {
  if (data.length === 0) {
    return (
      <div className="md:col-span-2">
        <EmptyChart title="Token Usage Over Time" />
      </div>
    );
  }
  const chartData = data.map((point) => ({
    date: formatBucketLabel(point.date),
    input: point.input_tokens,
    output: point.output_tokens,
    tokens: point.total_tokens,
    count: point.count,
  }));
  return (
    <Card className="md:col-span-2">
      <CardHeader>
        <CardTitle className="text-base">Token Usage Over Time</CardTitle>
      </CardHeader>
      <CardContent>
        <ChartContainer
          className="h-[300px]"
          config={{
            input: { label: "Input tokens", color: LINE_COLORS.input },
            output: { label: "Output tokens", color: LINE_COLORS.output },
            tokens: { label: "Tokens", color: LINE_COLORS.input },
            count: { label: "Requests", color: LINE_COLORS.output },
          }}
        >
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 12 }} />
            <ChartTooltip />
            <Legend />
            {embedding ? (
              <>
                <Line
                  type="monotone"
                  dataKey="tokens"
                  name="Tokens"
                  stroke="var(--color-tokens)"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="count"
                  name="Requests"
                  stroke="var(--color-count)"
                  strokeWidth={2}
                  dot={false}
                />
              </>
            ) : (
              <>
                <Line
                  type="monotone"
                  dataKey="input"
                  name="Input tokens"
                  stroke="var(--color-input)"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="output"
                  name="Output tokens"
                  stroke="var(--color-output)"
                  strokeWidth={2}
                  dot={false}
                />
              </>
            )}
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}

export function EmbeddingModelBars({ data }: { data: AnalysisModelStats[] }) {
  if (data.length === 0) {
    return (
      <div className="md:col-span-2">
        <EmptyChart title="Model Usage" />
      </div>
    );
  }
  return (
    <Card className="md:col-span-2">
      <CardHeader>
        <CardTitle className="text-base">Model Usage</CardTitle>
      </CardHeader>
      <CardContent>
        <ChartContainer
          className="h-[300px]"
          config={{ count: { label: "Requests", color: CHART_COLORS[0] } }}
        >
          <BarChart data={data} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" tick={{ fontSize: 12 }} />
            <YAxis
              dataKey="model"
              type="category"
              width={180}
              tick={{ fontSize: 12 }}
            />
            <ChartTooltip />
            <Bar dataKey="count" name="Requests" fill="var(--color-count)" />
          </BarChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}

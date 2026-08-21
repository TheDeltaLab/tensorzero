// Modified by Delta-AI under Apache 2.0
import { describe, expect, test } from "vitest";
import {
  computeCostCoverage,
  formatCostCoveragePercent,
  transformModelUsageData,
} from "./ModelUsage";
import type { ModelUsageTimePoint } from "~/types/tensorzero";

function row(
  overrides: Partial<ModelUsageTimePoint> &
    Pick<ModelUsageTimePoint, "period_start" | "model_name">,
): ModelUsageTimePoint {
  return {
    input_tokens: 10n,
    output_tokens: 4n,
    count: 100n,
    cost: null,
    count_with_cost: 0n,
    ...overrides,
  };
}

describe("computeCostCoverage", () => {
  test("ignores models that are not in the cost chart", () => {
    const rows = [
      row({
        period_start: "2026-08-20T12:00:00Z",
        model_name: "deepseek-v4-flash",
        count: 8n,
        cost: 0.000234,
        count_with_cost: 1n,
      }),
      row({
        period_start: "2026-08-20T12:00:00Z",
        model_name: "dummy::good",
        count: 543981n,
        count_with_cost: 0n,
      }),
    ];
    const {
      data: _data,
      modelNames,
      visiblePeriods,
    } = transformModelUsageData(rows, "cost");
    expect(modelNames).toEqual(["deepseek-v4-flash"]);
    const coverage = computeCostCoverage(rows, visiblePeriods, modelNames);
    expect(coverage).toEqual({ percent: 12.5, withCost: 1, total: 8 });
  });

  test("formats sub-percent coverage without rounding down to 0", () => {
    expect(formatCostCoveragePercent(0)).toBe("0");
    expect(formatCostCoveragePercent(0.18)).toBe("<1");
    expect(formatCostCoveragePercent(12.5)).toBe("13");
    expect(formatCostCoveragePercent(100)).toBe("100");
  });
});

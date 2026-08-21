// Modified by Delta-AI under Apache 2.0
import { describe, expect, test } from "vitest";
import { formatCost, normalizeCurrency } from "./cost";

describe("formatCost", () => {
  test("defaults to USD with a dollar sign", () => {
    expect(formatCost(0.0042)).toBe("$0.0042");
    expect(formatCost(0)).toBe("$0.00");
    expect(formatCost(1.5)).toBe("$1.50");
  });

  test("uses a yuan sign for CNY and RMB", () => {
    expect(formatCost(6, "CNY")).toBe("¥6.00");
    expect(formatCost(6, "rmb")).toBe("¥6.00");
  });

  test("falls back to a code suffix for unknown currencies", () => {
    expect(formatCost(1.25, "GBP")).toBe("1.25 GBP");
  });

  test("normalizeCurrency treats missing values as USD", () => {
    expect(normalizeCurrency(undefined)).toBe("USD");
    expect(normalizeCurrency("")).toBe("USD");
    expect(normalizeCurrency("RMB")).toBe("CNY");
  });
});

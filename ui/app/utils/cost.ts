// Modified by Delta-AI under Apache 2.0
/**
 * Format a cost amount with trailing zeros trimmed but always at least 2 decimal places.
 *
 * Examples:
 *   formatCost(0.0042)        → "$0.0042"
 *   formatCost(0)             → "$0.00"
 *   formatCost(1.5, "CNY")    → "¥1.50"
 *   formatCost(1.5, "GBP")    → "1.50 GBP"
 */
const CURRENCY_SYMBOLS: Record<string, string> = {
  USD: "$",
  CNY: "¥",
  EUR: "€",
  JPY: "¥",
};

export function normalizeCurrency(currency?: string | null): string {
  const code = (currency ?? "USD").trim().toUpperCase();
  if (!code) return "USD";
  if (code === "RMB") return "CNY";
  return code;
}

export function formatCost(cost: number, currency = "USD"): string {
  const code = normalizeCurrency(currency);
  const fixed = cost.toFixed(9);
  // Trim trailing zeros but keep at least 2 decimal places
  const trimmed = fixed.replace(/0+$/, "");
  const [integer, decimal = ""] = trimmed.split(".");
  const paddedDecimal = decimal.padEnd(2, "0");
  const amount = `${integer}.${paddedDecimal}`;
  const symbol = CURRENCY_SYMBOLS[code];
  if (symbol) return `${symbol}${amount}`;
  return `${amount} ${code}`;
}

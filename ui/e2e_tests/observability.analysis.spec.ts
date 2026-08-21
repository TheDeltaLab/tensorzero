// Modified by Delta-AI under Apache 2.0
import { test, expect } from "@playwright/test";

test("should show the analysis page", async ({ page }) => {
  await page.goto("/observability/analysis");
  await expect(page.getByRole("heading", { name: "Analysis" })).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Chat Analysis" }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Embedding Analysis" }),
  ).toBeVisible();
  await expect(page.getByRole("button", { name: "24h" })).toBeVisible();
  await expect(page.getByText("Input Cache Hit Rate")).toBeVisible();
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});

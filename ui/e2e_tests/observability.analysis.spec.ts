// Modified by Delta-AI under Apache 2.0
import { test, expect } from "@playwright/test";

test("should show the analysis page", async ({ page }) => {
  await page.goto("/observability/analysis");
  await expect(
    page.getByRole("heading", { name: "Analysis", exact: true }),
  ).toBeVisible();
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

test("should apply a custom time range", async ({ page }) => {
  await page.goto("/observability/analysis");
  const customButton = page.getByRole("button", { name: "Custom" });
  await expect(customButton).toBeVisible();
  await customButton.click();
  // The picker seeds a trailing 24h window, so both ends already have values.
  const picker = page.getByRole("dialog");
  await expect(picker.getByText("From", { exact: true })).toBeVisible();
  await expect(picker.getByText("To", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Apply" }).click();
  await page.waitForURL(/range=custom&from=.+&to=.+/);
  await expect(page.getByRole("button", { name: "Custom" })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});

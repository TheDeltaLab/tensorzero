// Modified by Delta-AI under Apache 2.0
import { test, expect } from "@playwright/test";
import type { Page } from "@playwright/test";

async function waitForTable(page: Page) {
  await expect(
    page
      .locator("tbody tr td")
      .first()
      .or(page.locator("tbody tr td").getByText("No inferences found")),
  ).toBeVisible();
}

test.describe("Inference Filtering", () => {
  test("should show Synapse-style filters above the table", async ({
    page,
  }) => {
    await page.goto("/observability/inferences");

    await expect(page.getByLabel("Provider")).toBeVisible();
    await expect(page.getByLabel("API key")).toBeVisible();
    await expect(page.getByLabel("Model")).toBeVisible();
    await expect(page.getByLabel("Cache")).toBeVisible();
    await expect(page.getByText("Time range")).toBeVisible();
    await expect(page.getByLabel("ID")).toBeVisible();
    await expect(page.getByLabel("Tags")).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "Filter" }),
    ).not.toBeVisible();
    await expect(
      page.getByRole("button", { name: "Apply Filters" }),
    ).not.toBeVisible();
  });

  test("should filter by function name from the top bar", async ({ page }) => {
    await page.goto("/observability/inferences");
    await waitForTable(page);

    await page.getByLabel("Function", { exact: true }).click();
    await page.getByRole("option", { name: "write_haiku" }).click();

    await expect(page).toHaveURL(/function_name=write_haiku/, {
      timeout: 10_000,
    });
    await waitForTable(page);
    await expect(page.getByLabel("Function write_haiku").first()).toBeVisible();
  });

  test("should filter by cache", async ({ page }) => {
    await page.goto("/observability/inferences");
    await waitForTable(page);

    const cacheField = page.getByLabel("Cache");
    await cacheField.click();
    await page.getByRole("option", { name: "Cached" }).click();

    await expect(page).toHaveURL(/cached=true/, { timeout: 10_000 });
  });

  test("should filter by api key", async ({ page }) => {
    await page.goto("/observability/inferences?api_key=abcdefghijkl");
    await waitForTable(page);
    await expect(page.getByLabel("API key")).toBeVisible();
    await page.getByLabel("API key").click();
    await page.getByRole("option", { name: "abcdefghijkl" }).click();
    await expect(page).toHaveURL(/api_key=abcdefghijkl/, { timeout: 10_000 });
    await page.getByRole("button", { name: "Clear", exact: true }).click();
    await expect(page).not.toHaveURL(/api_key=/, { timeout: 10_000 });
  });

  test("should clear function filter", async ({ page }) => {
    await page.goto("/observability/inferences?function_name=write_haiku");
    await waitForTable(page);
    await expect(page.getByLabel("Function write_haiku").first()).toBeVisible();

    await page.getByRole("button", { name: "Clear function filter" }).click();
    await expect(page).not.toHaveURL(/function_name/, { timeout: 10_000 });
  });

  test("should clear all top-bar filters", async ({ page }) => {
    await page.goto(
      "/observability/inferences?function_name=write_haiku&cached=true",
    );
    await expect(
      page.getByRole("button", { name: "Clear", exact: true }),
    ).toBeVisible();
    await page.getByRole("button", { name: "Clear", exact: true }).click();
    await expect(page).not.toHaveURL(/function_name/, { timeout: 10_000 });
    await expect(page).not.toHaveURL(/cached=/, { timeout: 10_000 });
  });

  test("should preserve filters when paginating", async ({ page }) => {
    await page.goto("/observability/inferences?function_name=write_haiku");
    await expect(page).toHaveURL(/function_name=write_haiku/);
    await waitForTable(page);

    const nextButton = page.getByRole("button", { name: /next/i });
    if (await nextButton.isVisible()) {
      const isDisabled = await nextButton.isDisabled();
      if (!isDisabled) {
        await nextButton.click();
        await expect(page).toHaveURL(/function_name=write_haiku/);
      }
    }
  });
});

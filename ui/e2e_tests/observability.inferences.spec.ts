// Modified by Delta-AI under Apache 2.0
import { test, expect } from "@playwright/test";

test("should show the inference list page", async ({ page }) => {
  await page.goto("/observability/inferences");
  await expect(page.getByRole("columnheader", { name: "Time" })).toBeVisible();
  await expect(
    page.getByRole("columnheader", { name: "Provider" }),
  ).toBeVisible();
  await expect(
    page.getByRole("columnheader", { name: "Status" }),
  ).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "TTFT" })).toBeVisible();
  await expect(
    page.getByRole("columnheader", { name: "Output tok/s" }),
  ).toBeVisible();

  // Assert that "error" is not in the page
  await expect(page.getByText("error", { exact: false })).not.toBeVisible();
});

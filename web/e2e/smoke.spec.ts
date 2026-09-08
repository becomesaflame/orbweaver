import { expect, test } from "@playwright/test";

test("web chat shell loads health version and composer", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("h1")).toHaveText("Orbweaver");
  await expect(page.locator("#app-ver")).toHaveText(/^v\d+\.\d+\.\d+$/);
  await expect(page.locator("#status")).toHaveText("disconnected");
  await expect(page.locator("#text")).toBeVisible();
  await expect(page.locator("#send")).toHaveText("Send");
  await expect(page.locator("#settings")).toBeVisible();
  await expect(page.locator("#token")).toBeVisible();
});

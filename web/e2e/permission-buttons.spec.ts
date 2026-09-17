import { expect, test, type Page } from "@playwright/test";

const SID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";

const REQUEST = {
  seq: 7,
  kind: "permission_request",
  payload: {
    tool_use_id: "tu-rm",
    name: "Bash",
    input: { command: "rm -rf build" },
    summary: "rm -rf build",
    reason: "destructive delete",
  },
};

async function renderRequest(page: Page) {
  await page.evaluate(
    ({ sid, ev }) => {
      const w = window as any;
      (document.getElementById("sid") as HTMLInputElement).value = sid;
      w.renderEvent(ev);
    },
    { sid: SID, ev: REQUEST },
  );
}

function card(page: Page) {
  return page.locator("#log .msg.permission[data-tool-use-id='tu-rm']");
}

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#status")).toHaveText("disconnected");
});

test("clicking Allow removes the approval buttons", async ({ page }) => {
  const approvals: object[] = [];
  await page.route("**/turns/approve", async (route) => {
    approvals.push(JSON.parse(route.request().postData() || "{}"));
    return route.fulfill({
      json: { status: "resolved", tool_use_id: "tu-rm", decision: "allow", scope: "once" },
    });
  });
  await renderRequest(page);
  const perm = card(page);
  await expect(perm.locator(".perm-actions button")).toHaveCount(3);
  await perm.locator("button.allow").first().click();
  // One decision is the whole choice: the buttons are gone, not just disabled.
  await expect(perm.locator(".perm-actions")).toHaveCount(0);
  await expect(perm.locator(".perm-outcome")).toContainText("Allowed");
  expect(approvals).toEqual([{ tool_use_id: "tu-rm", decision: "allow", scope: "once" }]);
});

test("clicking Deny removes the approval buttons", async ({ page }) => {
  await page.route("**/turns/approve", (route) =>
    route.fulfill({
      json: { status: "resolved", tool_use_id: "tu-rm", decision: "deny", scope: "once" },
    }),
  );
  await renderRequest(page);
  const perm = card(page);
  await perm.locator("button.deny").click();
  await expect(perm.locator(".perm-actions")).toHaveCount(0);
  await expect(perm.locator(".perm-outcome")).toContainText("Denied");
});

test("a decision made elsewhere removes the buttons here too", async ({ page }) => {
  await renderRequest(page);
  const perm = card(page);
  await expect(perm.locator(".perm-actions button")).toHaveCount(3);
  await page.evaluate(() => {
    (window as any).renderEvent({
      seq: 8,
      kind: "permission_response",
      payload: { tool_use_id: "tu-rm", name: "Bash", decision: "allow", scope: "session" },
    });
  });
  await expect(perm.locator(".perm-actions")).toHaveCount(0);
  await expect(perm.locator(".perm-outcome")).toContainText("Allowed for this session");
});

test("a failed decision puts the buttons back so the user can retry", async ({ page }) => {
  let calls = 0;
  await page.route("**/turns/approve", async (route) => {
    calls += 1;
    if (calls === 1) return route.fulfill({ status: 500, body: "boom" });
    return route.fulfill({
      json: { status: "resolved", tool_use_id: "tu-rm", decision: "allow", scope: "once" },
    });
  });
  await renderRequest(page);
  const perm = card(page);
  await perm.locator("button.allow").first().click();
  await expect(perm.locator(".perm-outcome")).toContainText("Could not send decision");
  await expect(perm.locator(".perm-actions button")).toHaveCount(3);
  await perm.locator("button.allow").first().click();
  await expect(perm.locator(".perm-actions")).toHaveCount(0);
  await expect(perm.locator(".perm-outcome")).toContainText("Allowed");
  expect(calls).toBe(2);
});

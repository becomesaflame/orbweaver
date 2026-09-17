import { expect, test, type Page } from "@playwright/test";

// Click-to-rename in the chat title bar (#chat-title -> #chat-title-edit).
const SID = "402e8376-ec10-4a76-960b-82b6d436c108";

function sessionRow(title: string) {
  return {
    id: SID,
    title,
    workspace_uri: "workspace:default",
    workspace_kind: "local",
    status: "active",
    channel: "web",
    model: "",
    created_at: "2026-09-16T00:00:00Z",
    last_event_at: "2026-09-16T00:00:00Z",
    event_count: 1,
    preview: title,
  };
}

/** Titles sent to PATCH /v1/sessions/{id}, newest last. */
async function mockGateway(page: Page, opts: { fail?: boolean } = {}) {
  const patched: string[] = [];
  await page.route("**/v1/**", async (route) => {
    const req = route.request();
    const url = req.url();
    const method = req.method();
    if (url.includes("/v1/models")) {
      return route.fulfill({ json: { models: [], defaults: { web: "test-model" } } });
    }
    if (url.replace(/\/$/, "").endsWith("/v1/sessions") && method === "GET") {
      return route.fulfill({ json: { sessions: [sessionRow("alpha chat")] } });
    }
    if (method === "PATCH" && /\/v1\/sessions\/[0-9a-f-]{36}$/i.test(url)) {
      const title = String(JSON.parse(req.postData() || "{}").title ?? "");
      patched.push(title);
      if (opts.fail) return route.fulfill({ status: 500, body: "rename boom" });
      return route.fulfill({ json: { id: SID, title } });
    }
    if (method === "GET" && /\/v1\/sessions\/[0-9a-f-]{36}\/events/i.test(url)) {
      return route.fulfill({
        json: { events: [{ seq: 1, kind: "user", payload: { text: "alpha prompt" } }] },
      });
    }
    return route.fulfill({ status: 404, body: `unmocked ${method} ${url}` });
  });
  return patched;
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem("orbweaver.jwt", "test-jwt");
  });
});

test("clicking the title bar renames the chat", async ({ page }) => {
  const patched = await mockGateway(page);
  await page.goto("/");
  await expect(page.locator("#chat-title")).toHaveText("alpha chat");
  await expect(page.locator("#chat-title-edit")).toBeHidden();

  await page.locator("#chat-title").click();
  const input = page.locator("#chat-title-edit");
  await expect(input).toBeVisible();
  await expect(input).toBeFocused();
  await expect(input).toHaveValue("alpha chat");
  await expect(page.locator("#chat-title")).toBeHidden();

  await input.fill("renamed chat");
  await input.press("Enter");

  await expect(page.locator("#chat-title")).toHaveText("renamed chat");
  await expect(input).toBeHidden();
  // The rail row follows without waiting for the next /v1/sessions poll.
  await expect(page.locator("button.chat .chat-title")).toHaveText("renamed chat");
  expect(patched).toEqual(["renamed chat"]);
});

test("Escape cancels the title edit without a PATCH", async ({ page }) => {
  const patched = await mockGateway(page);
  await page.goto("/");
  await expect(page.locator("#chat-title")).toHaveText("alpha chat");
  await page.locator("#chat-title").click();
  const input = page.locator("#chat-title-edit");
  await input.fill("discard me");
  await input.press("Escape");
  await expect(input).toBeHidden();
  await expect(page.locator("#chat-title")).toHaveText("alpha chat");
  expect(patched).toEqual([]);
});

test("blur commits a change; unchanged or blank titles send nothing", async ({ page }) => {
  const patched = await mockGateway(page);
  await page.goto("/");
  await expect(page.locator("#chat-title")).toHaveText("alpha chat");

  await page.locator("#chat-title").click();
  await page.locator("#text").click();
  await expect(page.locator("#chat-title")).toHaveText("alpha chat");
  expect(patched).toEqual([]);

  await page.locator("#chat-title").click();
  await page.locator("#chat-title-edit").fill("   ");
  await page.locator("#text").click();
  await expect(page.locator("#chat-title")).toHaveText("alpha chat");
  expect(patched).toEqual([]);

  await page.locator("#chat-title").click();
  await page.locator("#chat-title-edit").fill("blurred title");
  await page.locator("#text").click();
  await expect(page.locator("#chat-title")).toHaveText("blurred title");
  expect(patched).toEqual(["blurred title"]);
});

test("a failed rename restores the previous title and reports the error", async ({ page }) => {
  const patched = await mockGateway(page, { fail: true });
  await page.goto("/");
  await expect(page.locator("#chat-title")).toHaveText("alpha chat");
  await page.locator("#chat-title").click();
  await page.locator("#chat-title-edit").fill("nope");
  await page.locator("#chat-title-edit").press("Enter");
  await expect(page.locator("#status")).toContainText("rename boom");
  await expect(page.locator("#chat-title")).toHaveText("alpha chat");
  expect(patched).toEqual(["nope"]);
});

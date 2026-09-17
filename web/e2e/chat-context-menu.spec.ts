import { expect, test, type Page } from "@playwright/test";

// Right-click a sidebar chat row -> #chat-menu with Rename and Delete.
const SID_A = "402e8376-ec10-4a76-960b-82b6d436c108";
const SID_B = "5f1c9a20-1b2d-4c3e-8a9f-7d6e5c4b3a21";

function sessionRow(id: string, title: string) {
  return {
    id,
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

type Calls = { patched: string[]; deleted: string[] };

/**
 * Mock gateway. The session list is served from mutable state so a DELETE is
 * reflected by later polls, the way the real gateway hides a soft-deleted row.
 */
async function mockGateway(page: Page, opts: { failDelete?: boolean } = {}) {
  const calls: Calls = { patched: [], deleted: [] };
  let rows = [sessionRow(SID_A, "alpha chat"), sessionRow(SID_B, "beta chat")];
  await page.route("**/v1/**", async (route) => {
    const req = route.request();
    const url = req.url();
    const method = req.method();
    if (url.includes("/v1/models")) {
      return route.fulfill({ json: { models: [], defaults: { web: "test-model" } } });
    }
    if (url.replace(/\/$/, "").endsWith("/v1/sessions") && method === "GET") {
      return route.fulfill({ json: { sessions: rows } });
    }
    const idMatch = url.match(/\/v1\/sessions\/([0-9a-f-]{36})$/i);
    if (method === "PATCH" && idMatch) {
      const title = String(JSON.parse(req.postData() || "{}").title ?? "");
      calls.patched.push(title);
      rows = rows.map((r) => (r.id === idMatch[1] ? { ...r, title } : r));
      return route.fulfill({ json: { id: idMatch[1], title } });
    }
    if (method === "DELETE" && idMatch) {
      calls.deleted.push(idMatch[1]);
      if (opts.failDelete) return route.fulfill({ status: 409, body: "a turn is running" });
      rows = rows.filter((r) => r.id !== idMatch[1]);
      return route.fulfill({ json: { id: idMatch[1], status: "deleted" } });
    }
    if (method === "GET" && /\/v1\/sessions\/[0-9a-f-]{36}\/events/i.test(url)) {
      return route.fulfill({ json: { events: [{ seq: 1, kind: "user", payload: { text: "hi" } }] } });
    }
    return route.fulfill({ status: 404, body: `unmocked ${method} ${url}` });
  });
  return calls;
}

const rowByTitle = (page: Page, title: string) =>
  page.locator("button.chat", { has: page.locator(`.chat-title:text-is("${title}")`) });

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem("orbweaver.jwt", "test-jwt");
  });
});

test("right-click opens the menu for that row and Escape closes it", async ({ page }) => {
  await mockGateway(page);
  await page.goto("/");
  const menu = page.locator("#chat-menu");
  await expect(menu).toBeHidden();

  await rowByTitle(page, "beta chat").click({ button: "right" });
  await expect(menu).toBeVisible();
  // The menu names the row that was right-clicked, not the active chat.
  await expect(page.locator("#chat-menu-label")).toHaveText("beta chat");
  await expect(page.locator("#chat-menu-rename")).toBeVisible();
  await expect(page.locator("#chat-menu-delete")).toBeVisible();

  await page.keyboard.press("Escape");
  await expect(menu).toBeHidden();
});

test("clicking elsewhere dismisses the menu", async ({ page }) => {
  await mockGateway(page);
  await page.goto("/");
  await rowByTitle(page, "alpha chat").click({ button: "right" });
  await expect(page.locator("#chat-menu")).toBeVisible();
  await page.locator("#text").click();
  await expect(page.locator("#chat-menu")).toBeHidden();
});

test("Rename from the menu PATCHes and updates the rail", async ({ page }) => {
  const calls = await mockGateway(page);
  page.on("dialog", (d) => d.accept("renamed via menu"));
  await page.goto("/");

  await rowByTitle(page, "beta chat").click({ button: "right" });
  await page.locator("#chat-menu-rename").click();

  await expect(page.locator("#chat-menu")).toBeHidden();
  await expect(rowByTitle(page, "renamed via menu")).toBeVisible();
  expect(calls.patched).toEqual(["renamed via menu"]);
});

test("Delete removes a background chat and leaves the open one alone", async ({ page }) => {
  const calls = await mockGateway(page);
  page.on("dialog", (d) => d.accept());
  await page.goto("/");
  // alpha is the active chat; delete beta from the rail.
  await expect(page.locator("#chat-title")).toHaveText("alpha chat");

  await rowByTitle(page, "beta chat").click({ button: "right" });
  await page.locator("#chat-menu-delete").click();

  await expect(rowByTitle(page, "beta chat")).toHaveCount(0);
  await expect(rowByTitle(page, "alpha chat")).toBeVisible();
  await expect(page.locator("#chat-title")).toHaveText("alpha chat");
  expect(calls.deleted).toEqual([SID_B]);
});

test("deleting the open chat switches to the remaining one", async ({ page }) => {
  const calls = await mockGateway(page);
  page.on("dialog", (d) => d.accept());
  await page.goto("/");
  await expect(page.locator("#chat-title")).toHaveText("alpha chat");

  await rowByTitle(page, "alpha chat").click({ button: "right" });
  await page.locator("#chat-menu-delete").click();

  await expect(rowByTitle(page, "alpha chat")).toHaveCount(0);
  await expect(page.locator("#chat-title")).toHaveText("beta chat");
  expect(calls.deleted).toEqual([SID_A]);
});

test("dismissing the confirm deletes nothing", async ({ page }) => {
  const calls = await mockGateway(page);
  page.on("dialog", (d) => d.dismiss());
  await page.goto("/");

  await rowByTitle(page, "beta chat").click({ button: "right" });
  await page.locator("#chat-menu-delete").click();

  await expect(rowByTitle(page, "beta chat")).toBeVisible();
  expect(calls.deleted).toEqual([]);
});

test("a refused delete keeps the row and reports the error", async ({ page }) => {
  const calls = await mockGateway(page, { failDelete: true });
  page.on("dialog", (d) => d.accept());
  await page.goto("/");

  await rowByTitle(page, "beta chat").click({ button: "right" });
  await page.locator("#chat-menu-delete").click();

  await expect(page.locator("#status")).toContainText("a turn is running");
  await expect(rowByTitle(page, "beta chat")).toBeVisible();
  expect(calls.deleted).toEqual([SID_B]);
});

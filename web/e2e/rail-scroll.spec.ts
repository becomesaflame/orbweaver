import { expect, test, type Page } from "@playwright/test";

// The rail rebuilds #session-list on every /v1/sessions poll (3s while a turn
// runs, 8s idle) and on setActive()/rename/delete/search. Rebuilding threw away
// the scroll position, so reading further down the rail snapped back to the top
// a few seconds later. These tests scroll the rail and hold it across a repaint.

const WORKSPACES = 20;

function sessionRow(i: number) {
  const n = String(i).padStart(2, "0");
  return {
    id: `00000000-0000-4000-8000-${String(i).padStart(12, "0")}`,
    title: `chat ${n}`,
    workspace_uri: `workspace:ws${n}`,
    workspace_kind: "local",
    status: "active",
    channel: "web",
    model: "",
    // Distinct, increasing stamps pin the group order so the rail layout is
    // stable between repaints.
    created_at: `2026-09-${String((i % 28) + 1).padStart(2, "0")}T00:00:00Z`,
    last_event_at: `2026-09-${String((i % 28) + 1).padStart(2, "0")}T00:00:00Z`,
    event_count: 1,
    preview: `chat ${n}`,
    running: false,
  };
}

const ROWS = Array.from({ length: WORKSPACES }, (_, i) => sessionRow(i));

async function mockGateway(page: Page, polls: { count: number }) {
  await page.route("**/v1/**", async (route) => {
    const url = route.request().url();
    const method = route.request().method();
    if (url.includes("/v1/models")) {
      return route.fulfill({ json: { models: [], defaults: { web: "test-model" } } });
    }
    if (url.replace(/\/$/, "").endsWith("/v1/sessions") && method === "GET") {
      polls.count += 1;
      return route.fulfill({ json: { sessions: ROWS } });
    }
    const ev = url.match(/\/v1\/sessions\/([0-9a-f-]{36})\/events/i);
    if (ev && method === "GET") {
      return route.fulfill({
        json: { events: [{ seq: 1, kind: "assistant", payload: { text: "ready" } }] },
      });
    }
    return route.fulfill({ status: 404, body: `unmocked ${method} ${url}` });
  });
}

// Wait for at least one more /v1/sessions read, so the assertion is about a
// repaint that actually happened rather than a timer that has not fired yet.
async function afterPoll(page: Page, polls: { count: number }) {
  const before = polls.count;
  await expect.poll(() => polls.count, { timeout: 5000 }).toBeGreaterThan(before);
  // The read resolves into paintSessions() a microtask later; give the repaint
  // a frame to land before reading scrollTop.
  await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => r(null))));
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem("orbweaver.jwt", "test-jwt");
    // Poll the rail fast so a repaint is easy to trigger.
    localStorage.setItem("orbweaver.pollMs", "150");
  });
});

test("the rail keeps its scroll position across a poll repaint", async ({ page }) => {
  const polls = { count: 0 };
  await mockGateway(page, polls);
  await page.goto("/");
  await expect(page.locator("#session-list .chat").first()).toBeVisible();

  const list = page.locator("#session-list");
  await list.evaluate((n) => { n.scrollTop = 240; });
  expect(await list.evaluate((n) => n.scrollTop)).toBe(240);

  await afterPoll(page, polls);
  expect(await list.evaluate((n) => n.scrollTop)).toBe(240);

  // A second poll must not drift it either.
  await afterPoll(page, polls);
  expect(await list.evaluate((n) => n.scrollTop)).toBe(240);
});

test("the rail keeps its scroll position when a chat is selected", async ({ page }) => {
  const polls = { count: 0 };
  await mockGateway(page, polls);
  await page.goto("/");
  await expect(page.locator("#session-list .chat").first()).toBeVisible();

  const list = page.locator("#session-list");
  await list.evaluate((n) => { n.scrollTop = 200; });
  // SetActive() repaints the rail. Click a row near the bottom programmatically
  // so Playwright's scroll-into-view is not what moves the rail.
  await list.evaluate((n) => {
    const rows = n.querySelectorAll("button.chat");
    (rows[rows.length - 1] as HTMLButtonElement).click();
  });
  expect(await list.evaluate((n) => n.scrollTop)).toBe(200);
});
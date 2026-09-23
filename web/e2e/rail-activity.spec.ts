import { expect, test, type Page } from "@playwright/test";

// The rail shows which chats are working and which finished while you were
// somewhere else: `.chat.running` spins, `.chat.done` stays highlighted until
// the row is clicked. `running` on /v1/sessions is what makes a turn started
// outside this tab (Telegram, cron, a second browser) animate here.

const SID_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";
const SID_B = "402e8376-ec10-4a76-960b-82b6d436c108";

function sessionRow(id: string, title: string, over: Record<string, unknown> = {}) {
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
    running: false,
    ...over,
  };
}

const EVENTS: Record<string, object[]> = {
  [SID_A]: [
    { seq: 1, kind: "user", payload: { text: "alpha prompt" } },
    { seq: 2, kind: "assistant", payload: { text: "alpha reply" } },
  ],
  [SID_B]: [
    { seq: 1, kind: "user", payload: { text: "beta prompt" } },
    { seq: 2, kind: "assistant", payload: { text: "beta reply" } },
  ],
};

// The gateway state the mock serves. Tests mutate this between polls to stand
// in for a turn starting and finishing somewhere other than this browser.
type World = { running: Set<string>; lastEventAt: Record<string, string> };

async function mockGateway(page: Page, world: World) {
  await page.route("**/v1/**", async (route) => {
    const url = route.request().url();
    const method = route.request().method();
    if (url.includes("/v1/models")) {
      return route.fulfill({ json: { models: [], defaults: { web: "test-model" } } });
    }
    if (url.replace(/\/$/, "").endsWith("/v1/sessions") && method === "GET") {
      const rows = [SID_B, SID_A].map((sid) =>
        sessionRow(sid, sid === SID_A ? "alpha chat" : "beta chat", {
          running: world.running.has(sid),
          last_event_at: world.lastEventAt[sid] || "2026-09-16T00:00:00Z",
        }),
      );
      return route.fulfill({ json: { sessions: rows } });
    }
    const ev = url.match(/\/v1\/sessions\/([0-9a-f-]{36})\/events/i);
    if (ev && method === "GET") {
      return route.fulfill({ json: { events: EVENTS[ev[1]] || [] } });
    }
    return route.fulfill({ status: 404, body: `unmocked ${method} ${url}` });
  });
}

function chat(page: Page, title: string) {
  return page.locator("button.chat", { hasText: title });
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem("orbweaver.jwt", "test-jwt");
    // Poll the rail fast so these assertions do not wait on the 8s cadence.
    localStorage.setItem("orbweaver.pollMs", "150");
  });
});

test("a turn running elsewhere spins in the rail, then highlights when done", async ({
  page,
}) => {
  const world: World = { running: new Set(), lastEventAt: {} };
  await mockGateway(page, world);
  await page.goto("/");
  // Boot lands on beta (first row); alpha is the chat we are not watching.
  await expect(page.locator("#chat-title")).toHaveText("beta chat");
  await expect(chat(page, "alpha chat")).not.toHaveClass(/running/);

  // A turn starts on alpha from another channel: the rail's poll picks it up.
  world.running.add(SID_A);
  await expect(chat(page, "alpha chat")).toHaveClass(/running/);
  await expect(chat(page, "alpha chat")).not.toHaveClass(/done/);
  // The spinner element is the visible loop while it runs.
  await expect(chat(page, "alpha chat").locator(".chat-state .spin")).toBeVisible();

  // It finishes. Alpha is not the chat on screen, so it stays highlighted.
  world.running.delete(SID_A);
  world.lastEventAt[SID_A] = "2026-09-16T01:00:00Z";
  await expect(chat(page, "alpha chat")).toHaveClass(/done/);
  await expect(chat(page, "alpha chat")).not.toHaveClass(/running/);
  await expect(chat(page, "alpha chat").locator(".chat-state .spin")).toBeHidden();

  // The highlight survives repaints until the row is clicked.
  await page.locator("#search").fill("chat");
  await expect(chat(page, "alpha chat")).toHaveClass(/done/);

  await chat(page, "alpha chat").click();
  await expect(page.locator("#chat-title")).toHaveText("alpha chat");
  await expect(chat(page, "alpha chat")).not.toHaveClass(/done/);
});

test("the chat you are watching does not get a done highlight", async ({ page }) => {
  const world: World = { running: new Set([SID_B]), lastEventAt: {} };
  await mockGateway(page, world);
  await page.goto("/");
  await expect(page.locator("#chat-title")).toHaveText("beta chat");
  await expect(chat(page, "beta chat")).toHaveClass(/running/);

  world.running.delete(SID_B);
  world.lastEventAt[SID_B] = "2026-09-16T01:00:00Z";
  await expect(chat(page, "beta chat")).not.toHaveClass(/running/);
  // You watched it finish, so there is nothing to catch up on.
  await expect(chat(page, "beta chat")).not.toHaveClass(/done/);
});

test("a failed retry with no new events does not re-highlight a read chat", async ({
  page,
}) => {
  const world: World = {
    running: new Set(),
    lastEventAt: { [SID_A]: "2026-09-16T00:00:00Z" },
  };
  await page.addInitScript((sid) => {
    localStorage.setItem("orbweaver.seenAt", JSON.stringify([[sid, "2026-09-16T00:00:00Z"]]));
  }, SID_A);
  await mockGateway(page, world);
  await page.goto("/");
  await expect(page.locator("#chat-title")).toHaveText("beta chat");
  await expect(chat(page, "alpha chat")).not.toHaveClass(/done/);

  world.running.add(SID_A);
  await expect(chat(page, "alpha chat")).toHaveClass(/running/);

  world.running.delete(SID_A);
  await expect(chat(page, "alpha chat")).not.toHaveClass(/running/);
  await expect(chat(page, "alpha chat")).not.toHaveClass(/done/);
});

test("new events in a chat missed while the tab was away light it up", async ({ page }) => {
  // Nothing was ever seen running: the watermark from the previous visit is
  // what tells the rail alpha moved on.
  const world: World = { running: new Set(), lastEventAt: {} };
  await page.addInitScript(
    (sid) => {
      localStorage.setItem("orbweaver.seenAt", JSON.stringify([[sid, "2026-09-16T00:00:00Z"]]));
    },
    SID_A,
  );
  await mockGateway(page, world);
  await page.goto("/");
  await expect(page.locator("#chat-title")).toHaveText("beta chat");
  await expect(chat(page, "alpha chat")).not.toHaveClass(/done/);

  world.lastEventAt[SID_A] = "2026-09-16T02:00:00Z";
  await expect(chat(page, "alpha chat")).toHaveClass(/done/);
});

test("sending in this tab spins that chat's rail row right away", async ({ page }) => {
  const world: World = { running: new Set(), lastEventAt: {} };
  await mockGateway(page, world);
  await page.goto("/");
  await expect(page.locator("#chat-title")).toHaveText("beta chat");
  // A WebSocket that opens and never completes the turn keeps the flight open.
  await page.evaluate(() => {
    const w = window as unknown as { WebSocket: typeof WebSocket };
    class FakeWS {
      readyState = 0;
      onopen: ((ev?: object) => void) | null = null;
      onmessage: ((ev: { data: string }) => void) | null = null;
      onerror: (() => void) | null = null;
      onclose: ((ev: { code: number }) => void) | null = null;
      constructor() {
        queueMicrotask(() => {
          this.readyState = 1;
          if (this.onopen) this.onopen({});
        });
      }
      send() {}
      close() {}
    }
    w.WebSocket = FakeWS as unknown as typeof WebSocket;
  });
  await page.locator("#text").fill("work on it");
  await page.locator("#send").click();
  await expect(page.locator("#send")).toHaveText("Stop");
  // This tab's own flight drives the spinner without waiting for a poll.
  await expect(chat(page, "beta chat")).toHaveClass(/running/);
});

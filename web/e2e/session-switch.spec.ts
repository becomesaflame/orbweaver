import { expect, test, type Page } from "@playwright/test";

const SID_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";
const SID_B = "402e8376-ec10-4a76-960b-82b6d436c108";

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

const EVENTS: Record<string, object[]> = {
  [SID_A]: [
    { seq: 1, kind: "user", payload: { text: "alpha prompt" } },
    { seq: 2, kind: "assistant", payload: { text: "alpha reply" } },
  ],
  [SID_B]: [
    { seq: 1, kind: "user", payload: { text: "beta prompt" } },
    { seq: 2, kind: "assistant", payload: { text: "beta reply" } },
    {
      seq: 3,
      kind: "tool_call",
      payload: { id: "tu-read", name: "Read", input: { path: "agent.py" }, summary: "agent.py" },
    },
    {
      seq: 4,
      kind: "permission_decision",
      payload: {
        name: "Read",
        reason: "Classifier error — needs user approval",
        behavior: "ask",
        fast_path: "classifier",
        tool_use_id: "tu-read",
      },
    },
    {
      seq: 5,
      kind: "permission_request",
      payload: {
        name: "Read",
        input: { path: "agent.py" },
        reason: "Classifier error — needs user approval",
        summary: "agent.py",
        tool_use_id: "tu-read",
      },
    },
  ],
};

async function mockGateway(
  page: Page,
  opts: { delayA?: number; delayB?: number } = {},
) {
  await page.route("**/v1/**", async (route) => {
    const url = route.request().url();
    const method = route.request().method();
    if (url.includes("/v1/models")) {
      return route.fulfill({ json: { models: [], defaults: { web: "test-model" } } });
    }
    if (url.replace(/\/$/, "").endsWith("/v1/sessions") && method === "GET") {
      return route.fulfill({
        json: { sessions: [sessionRow(SID_B, "beta chat"), sessionRow(SID_A, "alpha chat")] },
      });
    }
    const ev = url.match(/\/v1\/sessions\/([0-9a-f-]{36})\/events/i);
    if (ev && method === "GET") {
      const sid = ev[1];
      const delay = sid === SID_A ? opts.delayA || 0 : sid === SID_B ? opts.delayB || 0 : 0;
      if (delay) await new Promise((r) => setTimeout(r, delay));
      return route.fulfill({ json: { events: EVENTS[sid] || [] } });
    }
    return route.fulfill({ status: 404, body: `unmocked ${method} ${url}` });
  });
}

function chat(page: Page, title: string) {
  return page.locator("button.chat", { hasText: title });
}

async function installFakeWebSocket(page: Page) {
  await page.evaluate(() => {
    const w = window as unknown as {
      __wsOpened: string[];
      __wsPayloads: string[];
      WebSocket: typeof WebSocket;
    };
    w.__wsOpened = [];
    w.__wsPayloads = [];
    class FakeWS {
      readyState = 0;
      onopen: ((ev?: object) => void) | null = null;
      onmessage: ((ev: { data: string }) => void) | null = null;
      onerror: (() => void) | null = null;
      onclose: ((ev: { code: number }) => void) | null = null;
      constructor(url: string) {
        w.__wsOpened.push(String(url));
        queueMicrotask(() => {
          this.readyState = 1;
          if (this.onopen) this.onopen({});
        });
      }
      send(data: string) {
        w.__wsPayloads.push(String(data));
      }
      close() {}
    }
    w.WebSocket = FakeWS as unknown as typeof WebSocket;
  });
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem("orbweaver.jwt", "test-jwt");
  });
});

test("clicking a chat replaces the transcript", async ({ page }) => {
  await mockGateway(page);
  await page.goto("/");
  await expect(page.locator("#log")).toContainText("beta prompt");
  await chat(page, "alpha chat").click();
  await expect(page.locator("#chat-title")).toHaveText("alpha chat");
  await expect(page.locator("#log")).toContainText("alpha prompt");
  await expect(page.locator("#log")).toContainText("alpha reply");
  await expect(page.locator("#log")).not.toContainText("beta prompt");
});

test("switching back mid-turn redraws the waiting chat", async ({ page }) => {
  await mockGateway(page);
  await page.goto("/");
  await expect(page.locator("#log")).toContainText("beta prompt");
  await installFakeWebSocket(page);
  await page.locator("#text").fill("keep going");
  await page.locator("#send").click();
  await expect(page.locator("#send")).toHaveText("Stop");
  await chat(page, "alpha chat").click();
  await expect(page.locator("#log")).toContainText("alpha prompt");
  await expect(page.locator("#log")).not.toContainText("beta prompt");
  await chat(page, "beta chat").click();
  await expect(page.locator("#chat-title")).toHaveText("beta chat");
  await expect(page.locator("#log")).toContainText("beta prompt");
  await expect(page.locator("#log")).toContainText("Approval needed");
  await expect(page.locator("#log")).not.toContainText("alpha prompt");
});

test("a turn in one chat does not block sending in another", async ({ page }) => {
  await mockGateway(page);
  await page.goto("/");
  await expect(page.locator("#log")).toContainText("beta prompt");
  await installFakeWebSocket(page);
  await page.locator("#text").fill("keep going");
  await page.locator("#send").click();
  await expect(page.locator("#send")).toHaveText("Stop");
  await chat(page, "alpha chat").click();
  await expect(page.locator("#chat-title")).toHaveText("alpha chat");
  await expect(page.locator("#send")).toHaveText("Send");
  await page.locator("#text").fill("hello from alpha");
  await page.locator("#send").click();
  await expect(page.locator("#text")).toHaveValue("");
  await expect(page.locator("#log")).toContainText("hello from alpha");
  await expect(page.locator("#send")).toHaveText("Stop");
  const opened = await page.evaluate(
    () => (window as unknown as { __wsOpened: string[] }).__wsOpened,
  );
  const payloads = await page.evaluate(
    () => (window as unknown as { __wsPayloads: string[] }).__wsPayloads,
  );
  expect(opened.some((url) => url.includes(SID_B))).toBeTruthy();
  expect(opened.some((url) => url.includes(SID_A))).toBeTruthy();
  expect(payloads.some((raw) => raw.includes("keep going"))).toBeTruthy();
  expect(payloads.some((raw) => raw.includes("hello from alpha"))).toBeTruthy();
});

test("a slower history fetch for the previous chat cannot overwrite the new one", async ({
  page,
}) => {
  await mockGateway(page, { delayA: 400, delayB: 40 });
  await page.goto("/");
  await expect(page.locator("#log")).toContainText("beta prompt");
  await chat(page, "alpha chat").click();
  await chat(page, "beta chat").click();
  await expect(page.locator("#chat-title")).toHaveText("beta chat");
  await page.waitForTimeout(500);
  await expect(page.locator("#log")).toContainText("beta prompt");
  await expect(page.locator("#log")).not.toContainText("alpha prompt");
});

test("hashchange loads the session in the URL", async ({ page }) => {
  await mockGateway(page);
  await page.goto("/");
  await expect(page.locator("#log")).toContainText("beta prompt");
  await page.evaluate((sid) => {
    location.hash = "#/s/" + sid;
  }, SID_A);
  await expect(page.locator("#chat-title")).toHaveText("alpha chat");
  await expect(page.locator("#log")).toContainText("alpha prompt");
  await expect(page.locator("#log")).not.toContainText("beta prompt");
});

import { expect, test, type Page } from "@playwright/test";

// The Continue button must appear only for a turn that really did not finish.
// Regression: the UI used to infer "stopped" from the kind of the last event,
// so bookkeeping that trails a finished turn (subagent_result, todo_state)
// made Continue show up on healthy chats after every reload / chat switch.
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
    event_count: 3,
    preview: title,
    running: false,
  };
}

async function mockGateway(
  page: Page,
  body: { events: object[]; last_turn_state?: string },
) {
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
    if (method === "GET" && /\/v1\/sessions\/[0-9a-f-]{36}\/events/i.test(url)) {
      return route.fulfill({ json: body });
    }
    return route.fulfill({ status: 404, body: `unmocked ${method} ${url}` });
  });
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem("orbweaver.jwt", "test-jwt");
    localStorage.setItem("orbweaver.sid", "402e8376-ec10-4a76-960b-82b6d436c108");
  });
});

test("no Continue button when bookkeeping trails a finished turn", async ({ page }) => {
  await mockGateway(page, {
    last_turn_state: "ok",
    events: [
      { seq: 1, kind: "user", payload: { text: "do the thing" } },
      { seq: 2, kind: "assistant", payload: { text: "done, here is the result" } },
      { seq: 3, kind: "subagent_result", payload: { subagent_id: "abc", status: "ok" } },
    ],
  });
  await page.goto("/");
  await expect(page.locator(".msg.assistant").first()).toContainText("here is the result");
  await expect(page.locator("#continue")).toBeHidden();
  await expect(page.locator(".msg.stopped")).toHaveCount(0);
});

test("Continue button shows for a turn that died mid-round", async ({ page }) => {
  await mockGateway(page, {
    last_turn_state: "stopped",
    events: [
      { seq: 1, kind: "user", payload: { text: "read the file" } },
      {
        seq: 2,
        kind: "tool_call",
        payload: { id: "tu-1", name: "Read", input: { path: "a.py" }, summary: "a.py" },
      },
      { seq: 3, kind: "tool_result", payload: { tool_use_id: "tu-1", content: "text" } },
    ],
  });
  await page.goto("/");
  await expect(page.locator("#continue")).toBeVisible();
});

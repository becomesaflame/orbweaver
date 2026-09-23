import { expect, test, type Page } from "@playwright/test";

// Workspace groups in the rail must stay put when a chat inside one of them
// gets new activity. Only the chats within a group may re-sort; the groups
// themselves are ordered by when the workspace was first used (its oldest
// chat), not by which one most recently had an event. Reordering groups on
// every poll made them hard to track — see the sidebar reorder regression.

const SID_OLD = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";
const SID_NEW = "402e8376-ec10-4a76-960b-82b6d436c108";

function sessionRow(id: string, title: string, over: Record<string, unknown> = {}) {
  return {
    id,
    title,
    status: "active",
    channel: "web",
    model: "",
    event_count: 1,
    preview: title,
    running: false,
    ...over,
  };
}

type World = { lastEventAt: Record<string, string> };

async function mockGateway(page: Page, world: World) {
  await page.route("**/v1/**", async (route) => {
    const url = route.request().url();
    const method = route.request().method();
    if (url.includes("/v1/models")) {
      return route.fulfill({ json: { models: [], defaults: { web: "test-model" } } });
    }
    if (url.replace(/\/$/, "").endsWith("/v1/sessions") && method === "GET") {
      // The gateway sorts by last_event_at descending. The workspace created
      // first ("workspace:old") has the older chat but, once it gets a new
      // event, its row would sort ahead of "workspace:new" here — that must
      // not drag the whole group ahead of workspace:new in the rail.
      const rows = [
        sessionRow(SID_NEW, "new-ws chat", {
          workspace_uri: "workspace:new",
          workspace_kind: "local",
          created_at: "2026-09-16T01:00:00Z",
          last_event_at: "2026-09-16T01:00:00Z",
        }),
        sessionRow(SID_OLD, "old-ws chat", {
          workspace_uri: "workspace:old",
          workspace_kind: "local",
          created_at: "2026-09-16T00:00:00Z",
          last_event_at: world.lastEventAt[SID_OLD] || "2026-09-16T00:00:00Z",
        }),
      ].sort((a, b) => (a.last_event_at < b.last_event_at ? 1 : -1));
      return route.fulfill({ json: { sessions: rows } });
    }
    const ev = url.match(/\/v1\/sessions\/([0-9a-f-]{36})\/events/i);
    if (ev && method === "GET") {
      return route.fulfill({ json: { events: [] } });
    }
    return route.fulfill({ status: 404, body: `unmocked ${method} ${url}` });
  });
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem("orbweaver.jwt", "test-jwt");
    localStorage.setItem("orbweaver.pollMs", "150");
  });
});

test("a workspace group does not jump ahead when its chat gets new activity", async ({
  page,
}) => {
  const world: World = { lastEventAt: {} };
  await mockGateway(page, world);
  await page.goto("/");

  const groupNames = () => page.locator(".ws-name").allTextContents();
  // workspace:old was created first, so it renders above workspace:new even
  // though workspace:new's chat currently has the more recent event.
  // wsLabel() strips the "workspace:" prefix for display.
  await expect.poll(groupNames).toEqual(["old", "new"]);

  // New activity lands on the older workspace's chat, making it the most
  // recently updated session overall. The group order must not change.
  world.lastEventAt[SID_OLD] = "2026-09-16T02:00:00Z";
  await page.waitForTimeout(300);
  await expect.poll(groupNames).toEqual(["old", "new"]);
});

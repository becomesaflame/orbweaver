import { expect, test, type Page } from "@playwright/test";

// Attaching files: drag-and-drop onto the stage, the paperclip picker, and
// paste. Uploads POST to /v1/sessions/{id}/uploads and the stored path is
// carried into the next turn's message.
const SID = "402e8376-ec10-4a76-960b-82b6d436c108";

type Calls = { uploads: string[] };

async function mockGateway(page: Page, opts: { uploadStatus?: number; uploadBody?: string } = {}) {
  const calls: Calls = { uploads: [] };
  const row = {
    id: SID,
    title: "alpha chat",
    workspace_uri: "workspace:default",
    workspace_kind: "local",
    status: "active",
    channel: "web",
    model: "",
    created_at: "2026-09-16T00:00:00Z",
    last_event_at: "2026-09-16T00:00:00Z",
    event_count: 1,
    preview: "alpha chat",
  };
  await page.route("**/v1/**", async (route) => {
    const req = route.request();
    const url = req.url();
    const method = req.method();
    if (url.includes("/v1/models")) {
      return route.fulfill({ json: { models: [], defaults: { web: "test-model" } } });
    }
    if (url.replace(/\/$/, "").endsWith("/v1/sessions") && method === "GET") {
      return route.fulfill({ json: { sessions: [row] } });
    }
    if (/\/uploads$/.test(url) && method === "POST") {
      // postData() is the raw multipart body; the filename appears in it.
      const body = req.postData() || "";
      const m = body.match(/filename="([^"]+)"/);
      const name = m ? m[1] : "unknown";
      calls.uploads.push(name);
      if (opts.uploadStatus && opts.uploadStatus >= 400) {
        return route.fulfill({ status: opts.uploadStatus, body: opts.uploadBody || "rejected" });
      }
      return route.fulfill({
        json: {
          path: "attachments/" + name,
          name,
          size: 12,
          kind: name.endsWith(".png") ? "image" : "text",
          truncated: false,
        },
      });
    }
    if (method === "GET" && /\/v1\/sessions\/[0-9a-f-]{36}\/events/i.test(url)) {
      return route.fulfill({ json: { events: [] } });
    }
    return route.fulfill({ status: 404, body: `unmocked ${method} ${url}` });
  });
  return calls;
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem("orbweaver.jwt", "test-jwt");
  });
});

// Turns run over a WebSocket, not POST /turns, so capture what the page sends
// on the socket. The fake opens, records, and never completes the turn.
async function captureTurnSocket(page: Page) {
  await page.evaluate(() => {
    const w = window as unknown as { WebSocket: typeof WebSocket; __sent: string[] };
    w.__sent = [];
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
      send(data: string) {
        w.__sent.push(String(data));
      }
      close() {}
    }
    w.WebSocket = FakeWS as unknown as typeof WebSocket;
  });
}

async function sentTurnText(page: Page): Promise<string> {
  const frames = await page.evaluate(
    () => (window as unknown as { __sent: string[] }).__sent || [],
  );
  // The turn payload is the frame carrying the message text.
  for (const raw of frames) {
    try {
      const msg = JSON.parse(raw);
      if (typeof msg.text === "string") return msg.text;
    } catch {
      /* non-JSON control frame */
    }
  }
  return "";
}

test("the paperclip picker uploads and shows a chip", async ({ page }) => {
  const calls = await mockGateway(page);
  await page.goto("/");

  await page.setInputFiles("#attach-input", {
    name: "notes.txt",
    mimeType: "text/plain",
    buffer: Buffer.from("hello from a text file"),
  });

  const chip = page.locator("#attach-strip .chip");
  await expect(chip).toHaveCount(1);
  await expect(chip.locator(".chip-name")).toHaveText("notes.txt");
  expect(calls.uploads).toEqual(["notes.txt"]);
});

test("dropping a file on the stage attaches it and shows the veil", async ({ page }) => {
  const calls = await mockGateway(page);
  await page.goto("/");
  await expect(page.locator("#drop-veil")).toBeHidden();

  // Build a DataTransfer in-page: Playwright cannot synthesize an OS drag.
  const handle = await page.evaluateHandle(() => {
    const dt = new DataTransfer();
    dt.items.add(new File(["col_a,col_b\n1,2\n"], "data.csv", { type: "text/csv" }));
    return dt;
  });
  await page.dispatchEvent("#stage", "dragenter", { dataTransfer: handle });
  await expect(page.locator("#drop-veil")).toBeVisible();

  await page.dispatchEvent("#stage", "drop", { dataTransfer: handle });
  await expect(page.locator("#drop-veil")).toBeHidden();
  await expect(page.locator("#attach-strip .chip")).toHaveCount(1);
  await expect(page.locator("#attach-strip .chip-name")).toHaveText("data.csv");
  expect(calls.uploads).toEqual(["data.csv"]);
});

test("attached paths ride along with the next message", async ({ page }) => {
  await mockGateway(page);
  await page.goto("/");
  await captureTurnSocket(page);

  await page.setInputFiles("#attach-input", {
    name: "report.pdf",
    mimeType: "application/pdf",
    buffer: Buffer.from("%PDF-1.4 fake"),
  });
  await expect(page.locator("#attach-strip .chip")).toHaveCount(1);

  await page.locator("#text").fill("summarise this");
  await page.locator("#send").click();

  await expect.poll(() => sentTurnText(page)).toContain("attachments/report.pdf");
  expect(await sentTurnText(page)).toContain("summarise this");
  // Chips clear once the turn is away.
  await expect(page.locator("#attach-strip .chip")).toHaveCount(0);
});

test("a chip can be removed before sending", async ({ page }) => {
  await mockGateway(page);
  await page.goto("/");
  await captureTurnSocket(page);

  await page.setInputFiles("#attach-input", {
    name: "scratch.txt",
    mimeType: "text/plain",
    buffer: Buffer.from("x"),
  });
  await expect(page.locator("#attach-strip .chip")).toHaveCount(1);
  await page.locator("#attach-strip .chip-x").click();
  await expect(page.locator("#attach-strip .chip")).toHaveCount(0);

  await page.locator("#text").fill("never mind");
  await page.locator("#send").click();
  await expect.poll(() => sentTurnText(page)).toContain("never mind");
  expect(await sentTurnText(page)).not.toContain("scratch.txt");
});

test("a rejected upload reports the error and does not reach the turn", async ({ page }) => {
  await mockGateway(page, { uploadStatus: 413, uploadBody: "file too large" });
  await page.goto("/");
  await captureTurnSocket(page);

  await page.setInputFiles("#attach-input", {
    name: "huge.bin",
    mimeType: "application/octet-stream",
    buffer: Buffer.from("0123456789"),
  });

  await expect(page.locator("#attach-strip .chip.failed")).toHaveCount(1);
  await expect(page.locator("#status")).toContainText("too large");

  await page.locator("#text").fill("look at this");
  await page.locator("#send").click();
  await expect.poll(() => sentTurnText(page)).toContain("look at this");
  expect(await sentTurnText(page)).not.toContain("huge.bin");
});

test("attachments alone can be sent with no typed text", async ({ page }) => {
  await mockGateway(page);
  await page.goto("/");
  await captureTurnSocket(page);

  await page.setInputFiles("#attach-input", {
    name: "shot.png",
    mimeType: "image/png",
    buffer: Buffer.from("\x89PNG\r\n\x1a\n fake"),
  });
  await expect(page.locator("#attach-strip .chip")).toHaveCount(1);

  await page.locator("#send").click();
  await expect.poll(() => sentTurnText(page)).toContain("attachments/shot.png");
});

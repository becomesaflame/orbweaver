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

test("renders markdown, tool cards and diffs from session events", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#status")).toHaveText("disconnected");
  // Vendored marked + DOMPurify are served by the /ui static mount.
  await expect
    .poll(() => page.evaluate(() => !!(window as any).marked && !!(window as any).DOMPurify))
    .toBe(true);

  // Drive the single rendering entry point with a fake turn: assistant Markdown,
  // a Read call+result and a StrReplace edit. Replaying a seq must be a no-op.
  await page.evaluate(() => {
    const w = window as any;
    const events = [
      { seq: 1, kind: "user", payload: { text: "fix the **bug**" } },
      {
        seq: 2,
        kind: "assistant",
        payload: { text: "## Plan\n\nRead [the file](https://example.com/app.py).\n\n```python\nprint(1)\n```" },
      },
      { seq: 3, kind: "tool_call", payload: { id: "tu-read", name: "Read", input: { path: "backend/app.py" }, summary: "backend/app.py" } },
      { seq: 4, kind: "permission_decision", payload: { tool_use_id: "tu-read", name: "Read", behavior: "allow", reason: "read", fast_path: "allow_read" } },
      { seq: 5, kind: "tool_result", payload: { tool_use_id: "tu-read", name: "Read", content: "1| import os\n2| x = 1" } },
      {
        seq: 6,
        kind: "tool_call",
        payload: { id: "tu-edit", name: "StrReplace", input: { path: "backend/app.py", old_string: "x = 1", new_string: "x = 2" }, summary: "backend/app.py" },
      },
      { seq: 7, kind: "tool_result", payload: { tool_use_id: "tu-edit", name: "StrReplace", content: "ok" } },
    ];
    for (const ev of events) w.renderEvent(ev);
    w.renderEvent(events[2]);
    w.renderEvent(events[4]);
  });

  const assistant = page.locator("#log .msg.assistant");
  await expect(assistant).toHaveCount(1);
  await expect(assistant.locator(".msg-body.md h2")).toHaveText("Plan");
  const link = assistant.locator("a[href='https://example.com/app.py']");
  await expect(link).toHaveAttribute("target", "_blank");
  await expect(link).toHaveAttribute("rel", /noopener/);
  await expect(assistant.locator(".codeblock .copy-btn")).toHaveText("Copy");
  await expect(page.locator("#log .msg.user strong")).toHaveText("bug");

  const cards = page.locator("#log .msg.card");
  await expect(cards).toHaveCount(2);
  const read = cards.nth(0);
  await expect(read).toHaveAttribute("data-tool-use-id", "tu-read");
  await expect(read.locator(".tool-name")).toHaveText("Read");
  await expect(read.locator(".tool-summary")).toHaveText("backend/app.py");
  await expect(read.locator(".chip")).toHaveText("done");
  await expect(read.locator(".card-body")).toBeHidden();
  await read.locator(".card-head").click();
  await expect(read.locator(".card-body")).toBeVisible();
  await expect(read.locator(".tool-output")).toContainText("import os");
  await expect(read.locator(".card-perm")).toContainText("Allowed");

  const edit = cards.nth(1);
  await expect(edit.locator(".tool-name")).toHaveText("StrReplace");
  await expect(edit.locator(".diff")).toBeVisible();
  await expect(edit.locator(".diff .dl.del")).toHaveText(/x = 1/);
  await expect(edit.locator(".diff .dl.add")).toHaveText(/x = 2/);
  await expect(edit.locator(".chip")).toHaveText("done");
});

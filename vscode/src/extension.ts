import * as vscode from "vscode";
import {
  ModelCatalog,
  SessionEvent,
  SessionRow,
  SessionSocket,
  cancelTurn,
  continueTurn,
  createSession,
  injectTurn,
  listEvents,
  listModels,
  listSessions,
  readConfig,
  renameSession,
  setSessionModel,
} from "./client";
import { DiffManager, PatchProposal } from "./diffs";
import { PlanManager } from "./plans";

class SessionTree implements vscode.TreeDataProvider<SessionRow> {
  private rows: SessionRow[] = [];
  private readonly _onChange = new vscode.EventEmitter<void>();
  readonly onDidChangeTreeData = this._onChange.event;

  constructor(private readonly getActive: () => string | undefined) {}

  refresh(rows: SessionRow[]): void {
    this.rows = rows;
    this._onChange.fire();
  }

  getTreeItem(row: SessionRow): vscode.TreeItem {
    const item = new vscode.TreeItem(row.title || "New chat", vscode.TreeItemCollapsibleState.None);
    item.id = row.id;
    item.description = row.preview || row.workspace_uri;
    item.tooltip = (row.workspace_uri || "") + (row.channel ? ` · ${row.channel}` : "");
    item.command = { command: "orbweaver.openSession", title: "Open", arguments: [row.id] };
    item.contextValue = "orbweaverSession";
    if (row.id === this.getActive()) item.iconPath = new vscode.ThemeIcon("comment-discussion");
    return item;
  }

  getChildren(): SessionRow[] {
    return this.rows;
  }
}

class ChatViewProvider implements vscode.WebviewViewProvider {
  private view: vscode.WebviewView | undefined;
  private busy = false;
  private stopped = false;

  private catalog: ModelCatalog | undefined;
  /** Session model override shown in the picker ("" = channel default). */
  private model = "";

  constructor(
    private readonly ctx: vscode.ExtensionContext,
    private readonly getSessionId: () => string | undefined,
    private readonly diffs: DiffManager,
    private readonly onModelPicked: (model: string) => Promise<void>
  ) {}

  resolveWebviewView(webviewView: vscode.WebviewView): void {
    this.view = webviewView;
    webviewView.webview.options = { enableScripts: true };
    webviewView.webview.html = chatHtml();
    webviewView.webview.onDidReceiveMessage(async (msg) => {
      const sid = this.getSessionId();
      try {
        if (msg.type === "ready") {
          this.post({ type: "patches", patches: this.diffs.list().map((p) => p.path) });
          await this.loadModels();
          return;
        }
        if (msg.type === "setModel") {
          const model = String(msg.model || "").trim();
          this.model = model;
          await this.onModelPicked(model);
          this.postModels();
          return;
        }
        if (msg.type === "send") {
          await vscode.commands.executeCommand("orbweaver.send", String(msg.text || ""));
          return;
        }
        if (msg.type === "stop" && sid) {
          await cancelTurn(sid, false);
          return;
        }
        if (msg.type === "inject" && sid) {
          const text = String(msg.text || "").trim();
          if (!text) return;
          const ev = await injectTurn(sid, text);
          this.post({ type: "event", event: ev });
          return;
        }
        if (msg.type === "continue" && sid) {
          await vscode.commands.executeCommand("orbweaver.continue");
          return;
        }
        if (msg.type === "accept") {
          await vscode.commands.executeCommand("orbweaver.acceptDiff", msg.path);
          return;
        }
        if (msg.type === "reject") {
          await vscode.commands.executeCommand("orbweaver.rejectDiff", msg.path);
          return;
        }
        if (msg.type === "openDiff") {
          await vscode.commands.executeCommand("orbweaver.openDiff", msg.path);
        }
      } catch (e) {
        this.post({ type: "error", text: String(e) });
      }
    });
  }

  setBusy(busy: boolean): void {
    this.busy = busy;
    this.post({ type: "busy", busy, stopped: this.stopped });
  }

  setStopped(stopped: boolean): void {
    this.stopped = stopped;
    this.post({ type: "busy", busy: this.busy, stopped });
  }

  showHistory(events: SessionEvent[]): void {
    this.post({ type: "history", events });
  }

  showEvent(event: SessionEvent): void {
    this.post({ type: "event", event });
  }

  showError(text: string): void {
    this.post({ type: "error", text });
  }

  setTitle(title: string, workspaceUri: string): void {
    this.post({ type: "meta", title, workspaceUri });
  }

  setPatches(paths: string[]): void {
    this.post({ type: "patches", patches: paths });
  }

  /** Current session's stored model ("" = gateway default for vscode). */
  setModel(model: string): void {
    this.model = model || "";
    this.postModels();
  }

  get pickedModel(): string {
    return this.model;
  }

  async loadModels(): Promise<void> {
    try {
      this.catalog = await listModels();
    } catch {
      /* no token yet or old gateway: the picker keeps only the current value */
    }
    this.postModels();
  }

  private postModels(): void {
    const defaults: Record<string, string> = this.catalog?.defaults || {};
    this.post({
      type: "models",
      models: this.catalog?.models || [],
      current: this.model,
      default: defaults.vscode || defaults.fallback || "",
    });
  }

  private post(msg: unknown): void {
    this.view?.webview.postMessage(msg);
  }
}

export function activate(context: vscode.ExtensionContext): void {
  const diffs = new DiffManager();
  const plans = new PlanManager(context);
  let sessionId: string | undefined = context.workspaceState.get("orbweaver.sessionId");
  let sessions: SessionRow[] = [];
  const tree = new SessionTree(() => sessionId);
  // Picker choice for a chat that does not exist yet; consumed by ensureSession.
  let pendingModel: string | undefined;
  const chat = new ChatViewProvider(context, () => sessionId, diffs, async (model) => {
    if (!sessionId) {
      pendingModel = model;
      return;
    }
    const stored = await setSessionModel(sessionId, model);
    const row = sessions.find((s) => s.id === sessionId);
    if (row) row.model = stored;
  });
  diffs.onPendingChange = (pending) => chat.setPatches(pending.map((p) => p.path));

  const socket = new SessionSocket(
    (ev) => {
      if (ev.kind === "subscribed") return;
      if (ev.kind === "turn_done") {
        chat.setBusy(false);
        chat.setStopped(ev.status === "stopped");
        refreshSessions().catch(() => undefined);
        return;
      }
      if (ev.kind === "error") {
        chat.showError(String(ev.detail || "turn error"));
        chat.setBusy(false);
        return;
      }
      chat.showEvent(ev);
      maybePatch(ev);
    },
    () => {
      /* reconnect on next send */
    }
  );

  async function refreshSessions(): Promise<SessionRow[]> {
    try {
      sessions = await listSessions();
      tree.refresh(sessions);
      return sessions;
    } catch (e) {
      tree.refresh([]);
      throw e;
    }
  }

  async function bindSession(id: string, opts: { history?: boolean } = {}): Promise<void> {
    sessionId = id;
    diffs.sessionId = id;
    await context.workspaceState.update("orbweaver.sessionId", id);
    const row = sessions.find((s) => s.id === id);
    chat.setTitle(row?.title || "Chat", row?.workspace_uri || readConfig().workspaceUri);
    chat.setModel(row?.model || "");
    tree.refresh(sessions);
    try {
      await socket.connect(id);
    } catch (e) {
      chat.showError(String(e));
    }
    if (opts.history !== false) {
      const events = await listEvents(id);
      chat.showHistory(events);
      chat.setStopped(events.length > 0 && events[events.length - 1].kind !== "assistant");
    }
  }

  async function ensureSession(): Promise<string> {
    if (sessionId) return sessionId;
    const created = await createSession(undefined, pendingModel);
    pendingModel = undefined;
    sessions = [created, ...sessions.filter((s) => s.id !== created.id)];
    tree.refresh(sessions);
    await bindSession(created.id, { history: false });
    chat.showHistory([]);
    return created.id;
  }

  function maybePatch(ev: SessionEvent): void {
    if (ev.kind !== "patch_proposal" && !(ev.kind === "tool_result" && ev.payload?.name === "ProposePatch")) {
      return;
    }
    try {
      const raw = ev.payload?.content ?? ev.payload;
      const parsed = typeof raw === "string" ? JSON.parse(raw) : raw;
      if (parsed && parsed.path) void diffs.applyProposal(parsed as PatchProposal);
    } catch {
      /* ignore */
    }
  }

  context.subscriptions.push(
    diffs,
    plans,
    vscode.workspace.registerTextDocumentContentProvider("orbweaver-diff", diffs),
    vscode.languages.registerCodeLensProvider({ scheme: "file" }, diffs),
    vscode.window.registerWebviewViewProvider("orbweaver.chat", chat),
    vscode.window.registerTreeDataProvider("orbweaver.sessions", tree),
    vscode.window.onDidChangeActiveTextEditor(() => diffs.refreshVisible()),
    vscode.commands.registerCommand("orbweaver.openChat", () =>
      vscode.commands.executeCommand("orbweaver.chat.focus")
    ),
    vscode.commands.registerCommand("orbweaver.newSession", async () => {
      try {
        // A new chat keeps the model picked in the view; else orbweaver.model.
        const created = await createSession(undefined, pendingModel ?? chat.pickedModel);
        pendingModel = undefined;
        sessions = [created, ...sessions.filter((s) => s.id !== created.id)];
        tree.refresh(sessions);
        await bindSession(created.id, { history: false });
        chat.showHistory([]);
        chat.setStopped(false);
        await vscode.commands.executeCommand("orbweaver.chat.focus");
      } catch (e) {
        vscode.window.showErrorMessage(String(e));
      }
    }),
    vscode.commands.registerCommand("orbweaver.refreshSessions", async () => {
      try {
        await refreshSessions();
        if (sessionId) chat.setTitle(
          sessions.find((s) => s.id === sessionId)?.title || "Chat",
          sessions.find((s) => s.id === sessionId)?.workspace_uri || readConfig().workspaceUri
        );
      } catch (e) {
        vscode.window.showErrorMessage(String(e));
      }
    }),
    vscode.commands.registerCommand("orbweaver.openSession", async (id: string) => {
      try {
        if (!sessions.length) await refreshSessions();
        await bindSession(id);
        await vscode.commands.executeCommand("orbweaver.chat.focus");
      } catch (e) {
        vscode.window.showErrorMessage(String(e));
      }
    }),
    vscode.commands.registerCommand("orbweaver.renameSession", async (row?: SessionRow) => {
      const id = row?.id || sessionId;
      if (!id) return;
      const current = sessions.find((s) => s.id === id);
      const next = await vscode.window.showInputBox({
        prompt: "Rename chat",
        value: current?.title || "",
      });
      if (!next?.trim()) return;
      await renameSession(id, next.trim());
      await refreshSessions();
    }),
    vscode.commands.registerCommand("orbweaver.send", async (text?: string) => {
      try {
        const sid = await ensureSession();
        const body = (text || "").trim();
        if (!body) return;
        chat.setBusy(true);
        chat.setStopped(false);
        chat.showEvent({ kind: "user", payload: { text: body } });
        if (socket.sessionId !== sid) await socket.connect(sid);
        socket.sendTurn(body, chat.pickedModel);
      } catch (e) {
        chat.setBusy(false);
        chat.showError(String(e));
      }
    }),
    vscode.commands.registerCommand("orbweaver.continue", async () => {
      const sid = sessionId;
      if (!sid) return;
      chat.setBusy(true);
      chat.setStopped(false);
      try {
        if (socket.sessionId !== sid) await socket.connect(sid);
        await continueTurn(sid);
      } catch (e) {
        chat.setBusy(false);
        chat.showError(String(e));
      }
    }),
    vscode.commands.registerCommand("orbweaver.acceptDiff", (filePath?: string) => diffs.accept(filePath)),
    vscode.commands.registerCommand("orbweaver.rejectDiff", (filePath?: string) => diffs.reject(filePath)),
    vscode.commands.registerCommand("orbweaver.openDiff", (filePath?: string) => diffs.openDiff(filePath)),
    vscode.commands.registerCommand("orbweaver.openPlan", () => plans.pickOrOpen()),
    vscode.commands.registerCommand("orbweaver.newPlan", () => plans.create("md")),
    vscode.commands.registerCommand("orbweaver.newHtmlPlan", () => plans.create("html")),
    vscode.commands.registerCommand("orbweaver.previewPlan", () => plans.showPreview())
  );

  refreshSessions()
    .then(async (rows) => {
      if (sessionId && rows.some((s) => s.id === sessionId)) {
        await bindSession(sessionId);
        return;
      }
      if (rows[0]) await bindSession(rows[0].id);
    })
    .catch((e) => chat.showError(String(e)));
}

export function deactivate() {}

function chatHtml(): string {
  return `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  :root { color-scheme: var(--vscode-editor-background) }
  body {
    font-family: var(--vscode-font-family);
    color: var(--vscode-foreground);
    margin: 0; display: flex; flex-direction: column; height: 100vh;
  }
  #meta { padding: 0.5rem 0.7rem 0.35rem; font-size: 0.75rem; opacity: 0.75; }
  #log { flex: 1; overflow: auto; padding: 0.4rem 0.7rem 0.8rem; }
  .msg { margin: 0.4rem 0; padding: 0.45rem 0.55rem; border-radius: 6px;
    background: var(--vscode-editor-background); border-left: 3px solid #888; white-space: pre-wrap; }
  .user { border-left-color: var(--vscode-button-background); }
  .assistant { border-left-color: #c4a574; }
  .tool { opacity: 0.8; font-size: 0.85em; }
  .error { border-left-color: #c05050; }
  #patches { padding: 0 0.7rem; }
  .patch { display: flex; gap: 0.35rem; align-items: center; margin: 0.25rem 0; font-size: 0.8rem; }
  .patch span { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  #composer { display: flex; flex-direction: column; gap: 0.35rem; padding: 0.55rem 0.7rem 0.8rem;
    border-top: 1px solid var(--vscode-widget-border); }
  textarea { width: 100%; min-height: 4rem; resize: vertical; box-sizing: border-box;
    background: var(--vscode-input-background); color: var(--vscode-input-foreground);
    border: 1px solid var(--vscode-input-border); padding: 0.4rem; }
  .row { display: flex; gap: 0.35rem; flex-wrap: wrap; }
  button { font: inherit; background: var(--vscode-button-background);
    color: var(--vscode-button-foreground); border: 0; padding: 0.3rem 0.55rem; border-radius: 4px; }
  button.ghost { background: transparent; color: var(--vscode-foreground);
    border: 1px solid var(--vscode-widget-border); }
  button:disabled { opacity: 0.45; }
  #model { font: inherit; font-size: 0.8rem; max-width: 12rem; margin-left: auto;
    background: var(--vscode-dropdown-background); color: var(--vscode-dropdown-foreground);
    border: 1px solid var(--vscode-dropdown-border); border-radius: 4px; padding: 0.2rem 0.3rem; }
</style>
</head>
<body>
  <div id="meta">No session</div>
  <div id="log"></div>
  <div id="patches"></div>
  <div id="composer">
    <textarea id="t" placeholder="Message (Enter to send, Shift+Enter newline)"></textarea>
    <div class="row">
      <button id="send">Send</button>
      <button id="stop" class="ghost" disabled>Stop</button>
      <button id="inject" class="ghost" disabled>Add to turn</button>
      <button id="cont" class="ghost" disabled>Continue</button>
      <select id="model" title="Model for this chat (default follows ORBWEAVER_VSCODE_MODEL)">
        <option value="">default model</option>
      </select>
    </div>
  </div>
<script>
const vscode = acquireVsCodeApi();
const log = document.getElementById("log");
const meta = document.getElementById("meta");
const patches = document.getElementById("patches");
const t = document.getElementById("t");
const modelSel = document.getElementById("model");
let busy = false;
let stopped = false;
function paintModels(m) {
  const want = String(m.current || "");
  modelSel.replaceChildren();
  const def = document.createElement("option");
  def.value = "";
  def.textContent = m.default ? "default (" + m.default + ")" : "default model";
  modelSel.appendChild(def);
  const seen = new Set();
  for (const row of m.models || []) {
    if (!row || !row.id || seen.has(row.id)) continue;
    seen.add(row.id);
    const o = document.createElement("option");
    o.value = row.id;
    o.textContent = row.available === false ? row.id + " (no key)" : row.id;
    o.disabled = row.available === false;
    modelSel.appendChild(o);
  }
  if (want && !seen.has(want)) {
    const o = document.createElement("option");
    o.value = want;
    o.textContent = want;
    modelSel.appendChild(o);
  }
  modelSel.value = want;
  if (modelSel.value !== want) modelSel.value = "";
}
modelSel.onchange = () => vscode.postMessage({ type: "setModel", model: modelSel.value });
function add(kind, text) {
  const d = document.createElement("div");
  d.className = "msg " + kind;
  d.textContent = (kind === "tool" || kind === "error" ? kind + ": " : "") + text;
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
}
function renderEvent(ev) {
  const p = ev.payload || {};
  if (ev.kind === "user" || ev.kind === "assistant") add(ev.kind, p.text || "");
  else if (ev.kind === "turn_aborted" || ev.kind === "cron_result") add("assistant", p.text || JSON.stringify(p));
  else if (ev.kind === "turn_interrupted") add("assistant", "Stopped");
  else if (ev.kind === "assistant_delta" || ev.kind === "tool_use_progress") return;
  else if (ev.kind === "tool_call") add("tool", (p.name || "tool") + " " + JSON.stringify(p.input || {}).slice(0, 240));
  else if (ev.kind === "patch_proposal") add("tool", "patch " + ((typeof p.content === "string" ? "" : (p.content || {}).path) || ""));
  else if (ev.kind === "subscribed" || ev.kind === "turn_done") return;
  else add("tool", ev.kind);
}
function paintPatches(list) {
  patches.replaceChildren();
  (list || []).forEach((path) => {
    const row = document.createElement("div");
    row.className = "patch";
    const name = document.createElement("span");
    name.textContent = path;
    const open = document.createElement("button");
    open.className = "ghost"; open.textContent = "Diff";
    open.onclick = () => vscode.postMessage({ type: "openDiff", path });
    const acc = document.createElement("button");
    acc.textContent = "Accept";
    acc.onclick = () => vscode.postMessage({ type: "accept", path });
    const rej = document.createElement("button");
    rej.className = "ghost"; rej.textContent = "Reject";
    rej.onclick = () => vscode.postMessage({ type: "reject", path });
    row.append(name, open, acc, rej);
    patches.appendChild(row);
  });
}
function sync() {
  document.getElementById("send").textContent = busy ? "…" : "Send";
  document.getElementById("send").disabled = busy;
  document.getElementById("stop").disabled = !busy;
  document.getElementById("inject").disabled = !busy || !t.value.trim();
  document.getElementById("cont").disabled = busy || !stopped;
}
document.getElementById("send").onclick = () => {
  const text = t.value.trim();
  if (!text || busy) return;
  vscode.postMessage({ type: "send", text });
  t.value = "";
  sync();
};
document.getElementById("stop").onclick = () => vscode.postMessage({ type: "stop" });
document.getElementById("inject").onclick = () => {
  const text = t.value.trim();
  if (!text) return;
  vscode.postMessage({ type: "inject", text });
  t.value = "";
  sync();
};
document.getElementById("cont").onclick = () => vscode.postMessage({ type: "continue" });
t.addEventListener("input", sync);
t.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.ctrlKey && !e.metaKey) {
    e.preventDefault();
    if (busy) document.getElementById("inject").click();
    else document.getElementById("send").click();
  }
});
window.addEventListener("message", (e) => {
  const m = e.data || {};
  if (m.type === "history") {
    log.replaceChildren();
    (m.events || []).forEach(renderEvent);
  } else if (m.type === "event") renderEvent(m.event);
  else if (m.type === "error") add("error", m.text || "error");
  else if (m.type === "busy") { busy = !!m.busy; stopped = !!m.stopped; sync(); }
  else if (m.type === "meta") meta.textContent = (m.title || "Chat") + " · " + (m.workspaceUri || "");
  else if (m.type === "patches") paintPatches(m.patches || []);
  else if (m.type === "models") paintModels(m);
});
vscode.postMessage({ type: "ready" });
sync();
</script>
</body></html>`;
}

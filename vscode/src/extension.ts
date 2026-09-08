import * as vscode from "vscode";
import * as http from "http";
import * as https from "https";
import * as path from "path";

interface PatchProposal {
  path: string;
  old: string;
  new: string;
  updated?: string;
}

let added: vscode.TextEditorDecorationType;
let removed: vscode.TextEditorDecorationType;
let lastPatch: { uri: vscode.Uri; proposal: PatchProposal } | undefined;
let sessionId: string | undefined;

function cfg() {
  const c = vscode.workspace.getConfiguration("orbweaver");
  return {
    gateway: String(c.get("gatewayUrl") || "http://127.0.0.1:8080").replace(/\/$/, ""),
    token: String(c.get("token") || ""),
    workspaceUri: String(c.get("workspaceUri") || "workspace:default"),
  };
}

function request(method: string, urlPath: string, body?: unknown): Promise<any> {
  const { gateway, token } = cfg();
  const u = new URL(gateway + urlPath);
  const payload = body === undefined ? undefined : JSON.stringify(body);
  const lib = u.protocol === "https:" ? https : http;
  return new Promise((resolve, reject) => {
    const req = lib.request(
      {
        hostname: u.hostname,
        port: u.port,
        path: u.pathname + u.search,
        method,
        headers: {
          "content-type": "application/json",
          ...(token ? { authorization: "Bearer " + token } : {}),
          ...(payload ? { "content-length": Buffer.byteLength(payload) } : {}),
        },
      },
      (res) => {
        const chunks: Buffer[] = [];
        res.on("data", (d) => chunks.push(d));
        res.on("end", () => {
          const text = Buffer.concat(chunks).toString("utf8");
          if ((res.statusCode || 500) >= 400) {
            reject(new Error(text));
            return;
          }
          resolve(text ? JSON.parse(text) : {});
        });
      }
    );
    req.on("error", reject);
    if (payload) req.write(payload);
    req.end();
  });
}

function workspaceUri(): string {
  const configured = cfg().workspaceUri.trim();
  if (configured) return configured;
  return "workspace:default";
}

async function ensureSession(): Promise<string> {
  if (sessionId) return sessionId;
  const { token } = cfg();
  if (!token) {
    throw new Error("Set orbweaver.token to a JWT from `python -m orbweaver.cli mint` on the gateway host");
  }
  const created = await request("POST", "/v1/sessions", {
    workspace_uri: workspaceUri(),
    workspace_kind: "local",
    title: vscode.workspace.name || "vscode",
    channel: "vscode",
  });
  sessionId = created.id;
  return sessionId!;
}

function applyDecorations(editor: vscode.TextEditor, proposal: PatchProposal) {
  const doc = editor.document.getText();
  const idx = proposal.old ? doc.indexOf(proposal.old) : -1;
  if (idx < 0) {
    const start = new vscode.Position(0, 0);
    editor.setDecorations(added, [
      { range: new vscode.Range(start, start), hoverMessage: "Orbweaver patch (old_string not in buffer)" },
    ]);
    return;
  }
  const start = editor.document.positionAt(idx);
  const end = editor.document.positionAt(idx + proposal.old.length);
  editor.setDecorations(removed, [{ range: new vscode.Range(start, end), hoverMessage: "deletion" }]);
  editor.setDecorations(added, [
    { range: new vscode.Range(end, end), hoverMessage: "addition:\n" + proposal.new },
  ]);
}

async function showPatch(proposal: PatchProposal) {
  const folder = vscode.workspace.workspaceFolders?.[0];
  if (!folder) return;
  const uri = vscode.Uri.joinPath(folder.uri, proposal.path);
  const doc = await vscode.workspace.openTextDocument(uri);
  const editor = await vscode.window.showTextDocument(doc);
  lastPatch = { uri, proposal };
  applyDecorations(editor, proposal);
}

class ChatViewProvider implements vscode.WebviewViewProvider {
  constructor(private readonly ctx: vscode.ExtensionContext) {}
  resolveWebviewView(webviewView: vscode.WebviewView) {
    webviewView.webview.options = { enableScripts: true };
    webviewView.webview.html = html();
    webviewView.webview.onDidReceiveMessage(async (msg) => {
      if (msg.type === "send") {
        try {
          const sid = await ensureSession();
          const r = await request("POST", `/v1/sessions/${sid}/turns`, { text: msg.text });
          webviewView.webview.postMessage({ type: "events", events: r.events });
          for (const ev of r.events || []) {
            if (ev.kind === "patch_proposal" || ev.kind === "tool_result") {
              try {
                const payload = ev.payload?.content || ev.payload;
                const parsed = typeof payload === "string" ? JSON.parse(payload) : payload;
                if (parsed && parsed.path && parsed.ok) await showPatch(parsed);
              } catch {
                /* ignore */
              }
            }
          }
        } catch (e) {
          webviewView.webview.postMessage({ type: "error", text: String(e) });
        }
      }
    });
  }
}

function html(): string {
  return `<!DOCTYPE html>
<html><body style="font-family:sans-serif;color:var(--vscode-foreground);">
<div id="log"></div>
<textarea id="t" style="width:100%;min-height:4rem;"></textarea>
<button id="s">Send</button>
<script>
const vscode = acquireVsCodeApi();
const log = document.getElementById("log");
document.getElementById("s").onclick = () => {
  const t = document.getElementById("t");
  vscode.postMessage({ type: "send", text: t.value });
  t.value = "";
};
window.addEventListener("message", (e) => {
  const m = e.data;
  const p = document.createElement("pre");
  if (m.type === "events" && Array.isArray(m.events)) {
    const bits = [];
    for (const ev of m.events) {
      const text = ev.payload && ev.payload.text;
      if (ev.kind === "assistant" || ev.kind === "turn_aborted") {
        bits.push((ev.kind === "turn_aborted" ? "aborted: " : "") + (text || JSON.stringify(ev.payload)));
      }
    }
    p.textContent = bits.length ? bits.join("\n\n") : JSON.stringify(m, null, 2);
  } else {
    p.textContent = JSON.stringify(m, null, 2);
  }
  log.appendChild(p);
});
</script>
</body></html>`;
}

export function activate(context: vscode.ExtensionContext) {
  added = vscode.window.createTextEditorDecorationType({
    backgroundColor: "rgba(80,180,80,0.25)",
    isWholeLine: true,
  });
  removed = vscode.window.createTextEditorDecorationType({
    backgroundColor: "rgba(180,80,80,0.25)",
    isWholeLine: true,
    textDecoration: "line-through",
  });
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider("orbweaver.chat", new ChatViewProvider(context)),
    vscode.commands.registerCommand("orbweaver.openChat", () =>
      vscode.commands.executeCommand("orbweaver.chat.focus")
    ),
    vscode.commands.registerCommand("orbweaver.acceptDiff", async () => {
      if (!lastPatch) return;
      const proposal = lastPatch.proposal;
      const edit = new vscode.WorkspaceEdit();
      const doc = await vscode.workspace.openTextDocument(lastPatch.uri);
      const text = doc.getText();
      const next = proposal.old && text.includes(proposal.old)
        ? text.replace(proposal.old, proposal.new)
        : proposal.updated || proposal.new;
      edit.replace(lastPatch.uri, new vscode.Range(doc.positionAt(0), doc.positionAt(text.length)), next);
      await vscode.workspace.applyEdit(edit);
      if (sessionId) {
        await request("POST", `/v1/sessions/${sessionId}/correction`, {
          text: "accepted patch for " + proposal.path,
          path: proposal.path,
          accepted: true,
        });
      }
      vscode.window.activeTextEditor?.setDecorations(added, []);
      vscode.window.activeTextEditor?.setDecorations(removed, []);
    }),
    vscode.commands.registerCommand("orbweaver.rejectDiff", async () => {
      if (!lastPatch || !sessionId) return;
      await request("POST", `/v1/sessions/${sessionId}/correction`, {
        text: "rejected patch for " + lastPatch.proposal.path,
        path: lastPatch.proposal.path,
        accepted: false,
      });
      vscode.window.activeTextEditor?.setDecorations(added, []);
      vscode.window.activeTextEditor?.setDecorations(removed, []);
    }),
    vscode.commands.registerCommand("orbweaver.openPlan", async () => {
      const uris = await vscode.window.showOpenDialog({
        canSelectMany: false,
        filters: { Plans: ["md", "html", "markdown"] },
      });
      if (!uris?.[0]) return;
      const doc = await vscode.workspace.openTextDocument(uris[0]);
      if (uris[0].path.endsWith(".html")) {
        const panel = vscode.window.createWebviewPanel(
          "orbweaver.plan",
          path.basename(uris[0].fsPath),
          vscode.ViewColumn.Beside,
          { enableScripts: true }
        );
        panel.webview.html = doc.getText();
      } else {
        await vscode.window.showTextDocument(doc);
      }
    })
  );
}

export function deactivate() {}

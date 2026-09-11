import * as http from "http";
import * as https from "https";
import * as vscode from "vscode";
import { JsonWebSocket } from "./ws";

export interface SessionRow {
  id: string;
  title: string;
  workspace_uri: string;
  workspace_kind?: string;
  status?: string;
  channel?: string;
  model?: string;
  created_at?: string;
  last_event_at?: string;
  event_count?: number;
  preview?: string;
}

export interface SessionEvent {
  id?: string;
  seq?: number;
  kind: string;
  payload?: Record<string, unknown>;
  status?: string;
  user_seq?: number;
  detail?: unknown;
}

export interface GatewayConfig {
  gateway: string;
  token: string;
  workspaceUri: string;
  /** Model for new chats; "" lets the gateway apply ORBWEAVER_VSCODE_MODEL. */
  model: string;
}

export interface ModelRow {
  id: string;
  provider?: string;
  available?: boolean;
  context_window?: number;
}

export interface ModelCatalog {
  models: ModelRow[];
  defaults: Record<string, string>;
}

export function readConfig(): GatewayConfig {
  const c = vscode.workspace.getConfiguration("orbweaver");
  return {
    gateway: String(c.get("gatewayUrl") || "http://127.0.0.1:8080").replace(/\/$/, ""),
    token: String(c.get("token") || ""),
    workspaceUri: String(c.get("workspaceUri") || "workspace:default").trim() || "workspace:default",
    model: String(c.get("model") || "").trim(),
  };
}

export function request(method: string, urlPath: string, body?: unknown): Promise<any> {
  const { gateway, token } = readConfig();
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
          "x-orbweaver-channel": "vscode",
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

export class SessionSocket {
  private sock = new JsonWebSocket();
  private sid = "";

  constructor(
    private readonly onEvent: (ev: SessionEvent) => void,
    private readonly onDisconnect: () => void
  ) {
    this.sock.onMessage = (data) => this.onEvent(data as unknown as SessionEvent);
    this.sock.onClose = () => this.onDisconnect();
    this.sock.onError = () => this.onDisconnect();
  }

  get sessionId(): string {
    return this.sid;
  }

  async connect(sessionId: string): Promise<void> {
    this.close();
    const { gateway, token } = readConfig();
    if (!token) throw new Error("Set orbweaver.token to a JWT from `python -m orbweaver.cli mint`");
    const u = new URL(gateway);
    u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
    u.pathname = `/v1/sessions/${sessionId}/ws`;
    u.search = "?token=" + encodeURIComponent(token);
    this.sid = sessionId;
    await this.sock.connect(u.toString());
    this.sock.send({ type: "subscribe" });
  }

  /** `model` (when set) is stored on the session by the gateway and used for this turn. */
  sendTurn(text: string, model?: string): void {
    const frame: Record<string, unknown> = { text };
    if (model) frame.model = model;
    this.sock.send(frame);
  }

  close(): void {
    this.sid = "";
    this.sock.close();
  }
}

export async function createSession(title?: string, model?: string): Promise<SessionRow> {
  const { token, workspaceUri, model: configured } = readConfig();
  if (!token) {
    throw new Error("Set orbweaver.token to a JWT from `python -m orbweaver.cli mint` on the gateway host");
  }
  // Picker choice first; the orbweaver.model setting seeds chats left on "default".
  const chosen = (model || configured).trim();
  const body: Record<string, unknown> = {
    workspace_uri: workspaceUri,
    workspace_kind: "local",
    title: title || vscode.workspace.name || "vscode",
    channel: "vscode",
  };
  if (chosen) body.model = chosen;
  const created = await request("POST", "/v1/sessions", body);
  return {
    id: created.id,
    title: created.title || title || "vscode",
    workspace_uri: created.workspace_uri || workspaceUri,
    channel: created.channel || "vscode",
    model: created.model || chosen || "",
  };
}

export async function listModels(): Promise<ModelCatalog> {
  const r = await request("GET", "/v1/models");
  return { models: (r.models || []) as ModelRow[], defaults: (r.defaults || {}) as Record<string, string> };
}

/** Empty `model` clears the session override so the channel default applies. */
export async function setSessionModel(sessionId: string, model: string): Promise<string> {
  const r = await request("PATCH", `/v1/sessions/${sessionId}`, { model });
  return String(r.model || "");
}

export async function listSessions(): Promise<SessionRow[]> {
  const r = await request("GET", "/v1/sessions");
  return (r.sessions || []) as SessionRow[];
}

export async function listEvents(sessionId: string): Promise<SessionEvent[]> {
  const r = await request("GET", `/v1/sessions/${sessionId}/events`);
  return (r.events || []) as SessionEvent[];
}

export async function renameSession(sessionId: string, title: string): Promise<void> {
  await request("PATCH", `/v1/sessions/${sessionId}`, { title });
}

export async function cancelTurn(sessionId: string, discard = false): Promise<void> {
  await request("POST", `/v1/sessions/${sessionId}/turns/cancel`, { discard });
}

export async function injectTurn(sessionId: string, text: string): Promise<SessionEvent> {
  const r = await request("POST", `/v1/sessions/${sessionId}/turns/inject`, { text });
  return r.event as SessionEvent;
}

export async function continueTurn(sessionId: string): Promise<any> {
  return request("POST", `/v1/sessions/${sessionId}/turns/continue`, {});
}

export async function sendCorrection(
  sessionId: string,
  path: string,
  accepted: boolean
): Promise<void> {
  await request("POST", `/v1/sessions/${sessionId}/correction`, {
    text: (accepted ? "accepted" : "rejected") + " patch for " + path,
    path,
    accepted,
  });
}

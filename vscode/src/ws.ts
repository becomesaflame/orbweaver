import * as crypto from "crypto";
import * as http from "http";
import * as https from "https";
import type { IncomingMessage } from "http";
import type { Socket } from "net";

export class JsonWebSocket {
  private socket: Socket | undefined;
  private buf: Buffer = Buffer.alloc(0);
  onMessage: (data: Record<string, unknown>) => void = () => {};
  onClose: () => void = () => {};
  onError: (err: Error) => void = () => {};

  connect(url: string): Promise<void> {
    this.close();
    const u = new URL(url);
    const key = crypto.randomBytes(16).toString("base64");
    const lib = u.protocol === "wss:" || u.protocol === "https:" ? https : http;
    const port = u.port ? Number(u.port) : u.protocol === "wss:" || u.protocol === "https:" ? 443 : 80;
    return new Promise((resolve, reject) => {
      const req = lib.request({
        hostname: u.hostname,
        port,
        path: u.pathname + u.search,
        method: "GET",
        headers: {
          Connection: "Upgrade",
          Upgrade: "websocket",
          "Sec-WebSocket-Version": "13",
          "Sec-WebSocket-Key": key,
        },
      });
      req.on("upgrade", (res: IncomingMessage, socket: Socket) => {
        if (res.statusCode !== 101) {
          socket.destroy();
          reject(new Error("websocket upgrade failed: " + res.statusCode));
          return;
        }
        this.socket = socket;
        socket.on("data", (chunk) => this.onChunk(chunk));
        socket.on("close", () => this.onClose());
        socket.on("error", (err) => this.onError(err));
        resolve();
      });
      req.on("error", reject);
      req.end();
    });
  }

  send(obj: unknown): void {
    if (!this.socket) throw new Error("websocket is not connected");
    const payload = Buffer.from(JSON.stringify(obj), "utf8");
    this.socket.write(encodeFrame(payload));
  }

  close(): void {
    if (!this.socket) return;
    try {
      this.socket.end();
    } catch {
      /* ignore */
    }
    this.socket = undefined;
    this.buf = Buffer.alloc(0);
  }

  private onChunk(chunk: Buffer): void {
    this.buf = Buffer.concat([this.buf, chunk]) as Buffer;
    while (true) {
      const parsed = decodeFrame(this.buf);
      if (!parsed) return;
      this.buf = parsed.rest;
      if (parsed.opcode === 8) {
        this.close();
        this.onClose();
        return;
      }
      if (parsed.opcode === 9) {
        this.socket?.write(encodeFrame(parsed.payload, 0xa));
        continue;
      }
      if (parsed.opcode !== 1 && parsed.opcode !== 0) continue;
      try {
        const data = JSON.parse(parsed.payload.toString("utf8"));
        if (data && typeof data === "object") this.onMessage(data as Record<string, unknown>);
      } catch {
        /* ignore malformed frames */
      }
    }
  }
}

function encodeFrame(payload: Buffer, opcode = 0x1): Buffer {
  const mask = crypto.randomBytes(4);
  const masked = Buffer.alloc(payload.length);
  for (let i = 0; i < payload.length; i++) masked[i] = payload[i] ^ mask[i % 4];
  let header: Buffer;
  if (payload.length < 126) {
    header = Buffer.alloc(6);
    header[1] = 0x80 | payload.length;
    mask.copy(header, 2);
  } else if (payload.length < 65536) {
    header = Buffer.alloc(8);
    header[1] = 0x80 | 126;
    header.writeUInt16BE(payload.length, 2);
    mask.copy(header, 4);
  } else {
    header = Buffer.alloc(14);
    header[1] = 0x80 | 127;
    header.writeUInt32BE(0, 2);
    header.writeUInt32BE(payload.length, 6);
    mask.copy(header, 10);
  }
  header[0] = 0x80 | opcode;
  return Buffer.concat([header, masked]);
}

function decodeFrame(buf: Buffer): { opcode: number; payload: Buffer; rest: Buffer } | undefined {
  if (buf.length < 2) return undefined;
  const opcode = buf[0] & 0x0f;
  const masked = (buf[1] & 0x80) !== 0;
  let len = buf[1] & 0x7f;
  let offset = 2;
  if (len === 126) {
    if (buf.length < 4) return undefined;
    len = buf.readUInt16BE(2);
    offset = 4;
  } else if (len === 127) {
    if (buf.length < 10) return undefined;
    len = buf.readUInt32BE(6);
    offset = 10;
  }
  const maskLen = masked ? 4 : 0;
  if (buf.length < offset + maskLen + len) return undefined;
  let payload = buf.subarray(offset + maskLen, offset + maskLen + len);
  if (masked) {
    const mask = buf.subarray(offset, offset + 4);
    const out = Buffer.alloc(len);
    for (let i = 0; i < len; i++) out[i] = payload[i] ^ mask[i % 4];
    payload = out;
  }
  return { opcode, payload, rest: buf.subarray(offset + maskLen + len) };
}

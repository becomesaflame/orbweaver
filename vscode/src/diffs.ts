import * as path from "path";
import * as vscode from "vscode";
import { sendCorrection } from "./client";

export interface PatchProposal {
  path: string;
  old?: string;
  new?: string;
  updated?: string;
  ok?: boolean;
  error?: string;
}

export interface PendingPatch {
  path: string;
  uri: vscode.Uri;
  original: string;
  proposed: string;
}

const SCHEME = "orbweaver-diff";

export class DiffManager implements vscode.TextDocumentContentProvider, vscode.CodeLensProvider {
  private readonly originals = new Map<string, string>();
  private readonly pending = new Map<string, PendingPatch>();
  private readonly _onDidChange = new vscode.EventEmitter<vscode.Uri>();
  private readonly _onDidChangeCodeLenses = new vscode.EventEmitter<void>();
  private added: vscode.TextEditorDecorationType;
  private removed: vscode.TextEditorDecorationType;
  readonly onDidChange = this._onDidChange.event;
  readonly onDidChangeCodeLenses = this._onDidChangeCodeLenses.event;
  sessionId: string | undefined;
  onPendingChange: (patches: PendingPatch[]) => void = () => {};

  constructor() {
    this.added = vscode.window.createTextEditorDecorationType({
      backgroundColor: "rgba(80,180,80,0.22)",
      isWholeLine: true,
      overviewRulerColor: "rgba(80,180,80,0.7)",
      overviewRulerLane: vscode.OverviewRulerLane.Right,
    });
    this.removed = vscode.window.createTextEditorDecorationType({
      backgroundColor: "rgba(180,80,80,0.18)",
      isWholeLine: true,
      overviewRulerColor: "rgba(180,80,80,0.7)",
      overviewRulerLane: vscode.OverviewRulerLane.Left,
    });
  }

  dispose(): void {
    this.added.dispose();
    this.removed.dispose();
    this._onDidChange.dispose();
    this._onDidChangeCodeLenses.dispose();
  }

  list(): PendingPatch[] {
    return [...this.pending.values()];
  }

  provideTextDocumentContent(uri: vscode.Uri): string {
    return this.originals.get(uri.toString()) || "";
  }

  provideCodeLenses(document: vscode.TextDocument): vscode.CodeLens[] {
    const patch = this.pending.get(document.uri.toString());
    if (!patch) return [];
    const range = new vscode.Range(0, 0, 0, 0);
    return [
      new vscode.CodeLens(range, {
        title: "Accept patch",
        command: "orbweaver.acceptDiff",
        arguments: [patch.path],
      }),
      new vscode.CodeLens(range, {
        title: "Reject patch",
        command: "orbweaver.rejectDiff",
        arguments: [patch.path],
      }),
      new vscode.CodeLens(range, {
        title: "Open diff",
        command: "orbweaver.openDiff",
        arguments: [patch.path],
      }),
    ];
  }

  async applyProposal(proposal: PatchProposal): Promise<void> {
    if (!proposal.path || proposal.ok === false) {
      vscode.window.showWarningMessage(
        "Orbweaver patch failed: " + (proposal.error || proposal.path || "unknown")
      );
      return;
    }
    const folder = vscode.workspace.workspaceFolders?.[0];
    if (!folder) return;
    const uri = vscode.Uri.joinPath(folder.uri, proposal.path);
    let original = "";
    try {
      const current = await vscode.workspace.openTextDocument(uri);
      original = current.getText();
    } catch {
      original = "";
    }
    const proposed =
      proposal.updated !== undefined
        ? proposal.updated
        : proposal.old && original.includes(proposal.old)
          ? original.replace(proposal.old, proposal.new || "")
          : proposal.new || original;
    const originalUri = this.originalUri(proposal.path);
    this.originals.set(originalUri.toString(), original);
    this._onDidChange.fire(originalUri);
    const patch: PendingPatch = { path: proposal.path, uri, original, proposed };
    this.pending.set(uri.toString(), patch);
    const doc = await this.ensureDocument(uri, original === "" && proposed !== "");
    const edit = new vscode.WorkspaceEdit();
    edit.replace(uri, new vscode.Range(doc.positionAt(0), doc.positionAt(doc.getText().length)), proposed);
    await vscode.workspace.applyEdit(edit);
    this.decorate(uri, original, proposed);
    this._onDidChangeCodeLenses.fire();
    this.onPendingChange(this.list());
    await this.openDiff(proposal.path);
  }

  async openDiff(filePath?: string): Promise<void> {
    const patch = this.patchByPath(filePath);
    if (!patch) return;
    const originalUri = this.originalUri(patch.path);
    await vscode.commands.executeCommand(
      "vscode.diff",
      originalUri,
      patch.uri,
      path.basename(patch.path) + " (Orbweaver)",
      { preview: false }
    );
  }

  async accept(filePath?: string): Promise<void> {
    const patch = this.patchByPath(filePath);
    if (!patch) return;
    const doc = await vscode.workspace.openTextDocument(patch.uri);
    if (doc.isDirty) await doc.save();
    await this.finish(patch, true);
  }

  async reject(filePath?: string): Promise<void> {
    const patch = this.patchByPath(filePath);
    if (!patch) return;
    const doc = await vscode.workspace.openTextDocument(patch.uri);
    const edit = new vscode.WorkspaceEdit();
    edit.replace(
      patch.uri,
      new vscode.Range(doc.positionAt(0), doc.positionAt(doc.getText().length)),
      patch.original
    );
    await vscode.workspace.applyEdit(edit);
    if (patch.original === "") {
      /* leave empty new file unsaved so reject can drop it */
    } else if (doc.isDirty) {
      await doc.save();
    }
    await this.finish(patch, false);
  }

  private async finish(patch: PendingPatch, accepted: boolean): Promise<void> {
    this.pending.delete(patch.uri.toString());
    this.clearDecorations(patch.uri);
    this._onDidChangeCodeLenses.fire();
    this.onPendingChange(this.list());
    if (this.sessionId) {
      try {
        await sendCorrection(this.sessionId, patch.path, accepted);
      } catch (e) {
        vscode.window.showErrorMessage(String(e));
      }
    }
  }

  private async ensureDocument(uri: vscode.Uri, create: boolean): Promise<vscode.TextDocument> {
    if (create) {
      const edit = new vscode.WorkspaceEdit();
      edit.createFile(uri, { ignoreIfExists: true });
      await vscode.workspace.applyEdit(edit);
    }
    try {
      return await vscode.workspace.openTextDocument(uri);
    } catch {
      const edit = new vscode.WorkspaceEdit();
      edit.createFile(uri, { ignoreIfExists: true });
      await vscode.workspace.applyEdit(edit);
      return vscode.workspace.openTextDocument(uri);
    }
  }

  private patchByPath(filePath?: string): PendingPatch | undefined {
    if (filePath) {
      for (const p of this.pending.values()) {
        if (p.path === filePath) return p;
      }
    }
    const active = vscode.window.activeTextEditor?.document.uri.toString();
    if (active && this.pending.has(active)) return this.pending.get(active);
    return this.pending.values().next().value;
  }

  private originalUri(filePath: string): vscode.Uri {
    return vscode.Uri.from({ scheme: SCHEME, path: "/" + filePath });
  }

  private decorate(uri: vscode.Uri, original: string, proposed: string): void {
    const editor = vscode.window.visibleTextEditors.find((e) => e.document.uri.toString() === uri.toString());
    if (!editor) return;
    const { added, removed } = lineDecorations(original, proposed, editor.document);
    editor.setDecorations(this.added, added);
    editor.setDecorations(this.removed, removed);
  }

  private clearDecorations(uri: vscode.Uri): void {
    const editor = vscode.window.visibleTextEditors.find((e) => e.document.uri.toString() === uri.toString());
    editor?.setDecorations(this.added, []);
    editor?.setDecorations(this.removed, []);
  }

  refreshVisible(): void {
    for (const patch of this.pending.values()) {
      this.decorate(patch.uri, patch.original, patch.proposed);
    }
  }
}

function lineDecorations(
  original: string,
  proposed: string,
  doc: vscode.TextDocument
): { added: vscode.DecorationOptions[]; removed: vscode.DecorationOptions[] } {
  const oldLines = original.split("\n");
  const newLines = proposed.split("\n");
  const added: vscode.DecorationOptions[] = [];
  const removed: vscode.DecorationOptions[] = [];
  const oldSet = new Set(oldLines);
  const newSet = new Set(newLines);
  for (let i = 0; i < newLines.length; i++) {
    if (!oldSet.has(newLines[i])) {
      const line = Math.min(i, Math.max(doc.lineCount - 1, 0));
      added.push({ range: doc.lineAt(line).range, hoverMessage: "Orbweaver added" });
    }
  }
  for (let i = 0; i < oldLines.length; i++) {
    if (!newSet.has(oldLines[i])) {
      const line = Math.min(i, Math.max(doc.lineCount - 1, 0));
      removed.push({
        range: doc.lineAt(line).range,
        hoverMessage: "removed:\n" + oldLines[i],
      });
    }
  }
  return { added, removed };
}

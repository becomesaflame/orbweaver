import * as path from "path";
import * as vscode from "vscode";

const PLAN_GLOB = "{**/.orbweaver/plan.md,**/.orbweaver/*.md,**/plans/**/*.md,**/plans/**/*.html,**/*.plan.md}";

export class PlanManager {
  private current: vscode.Uri | undefined;
  private preview: vscode.WebviewPanel | undefined;
  private watcher: vscode.FileSystemWatcher | undefined;

  constructor(private readonly ctx: vscode.ExtensionContext) {
    const saved = this.ctx.workspaceState.get<string>("orbweaver.planUri");
    if (saved) this.current = vscode.Uri.parse(saved);
  }

  get uri(): vscode.Uri | undefined {
    return this.current;
  }

  async pickOrOpen(): Promise<void> {
    const existing = await this.findPlans();
    if (this.current) {
      await this.open(this.current);
      return;
    }
    if (existing.length === 1) {
      await this.open(existing[0]);
      return;
    }
    if (existing.length > 1) {
      const picked = await vscode.window.showQuickPick(
        existing.map((u) => ({ label: vscode.workspace.asRelativePath(u), uri: u })),
        { placeHolder: "Open a living plan" }
      );
      if (picked) await this.open(picked.uri);
      return;
    }
    await this.create("md");
  }

  async create(kind: "md" | "html" = "md"): Promise<void> {
    const folder = vscode.workspace.workspaceFolders?.[0];
    if (!folder) {
      vscode.window.showErrorMessage("Open a folder to keep a living plan.");
      return;
    }
    const stamp = new Date().toISOString().slice(0, 10);
    const rel = kind === "html" ? `.orbweaver/plan-${stamp}.html` : `.orbweaver/plan.md`;
    const uri = vscode.Uri.joinPath(folder.uri, rel);
    try {
      await vscode.workspace.fs.stat(uri);
    } catch {
      const seed =
        kind === "html"
          ? `<!DOCTYPE html><html><body><h1>Plan</h1><p>${stamp}</p></body></html>\n`
          : `# Plan\n\n${stamp}\n\n## Now\n\n- \n\n## Next\n\n- \n`;
      await vscode.workspace.fs.createDirectory(vscode.Uri.joinPath(folder.uri, ".orbweaver"));
      await vscode.workspace.fs.writeFile(uri, Buffer.from(seed, "utf8"));
    }
    await this.open(uri);
  }

  async open(uri: vscode.Uri): Promise<void> {
    this.current = uri;
    await this.ctx.workspaceState.update("orbweaver.planUri", uri.toString());
    const doc = await vscode.workspace.openTextDocument(uri);
    await vscode.window.showTextDocument(doc, { preview: false, viewColumn: vscode.ViewColumn.One });
    await this.showPreview(doc);
    this.watch(uri);
  }

  async showPreview(doc?: vscode.TextDocument): Promise<void> {
    const uri = doc?.uri || this.current;
    if (!uri) return;
    const openDoc = doc || (await vscode.workspace.openTextDocument(uri));
    if (uri.path.endsWith(".md") || uri.path.endsWith(".markdown")) {
      await vscode.commands.executeCommand("markdown.showPreviewToSide", uri);
      return;
    }
    if (uri.path.endsWith(".html") || uri.path.endsWith(".htm")) {
      if (!this.preview) {
        this.preview = vscode.window.createWebviewPanel(
          "orbweaver.plan",
          path.basename(uri.fsPath),
          vscode.ViewColumn.Beside,
          { enableScripts: true, retainContextWhenHidden: true }
        );
        this.preview.onDidDispose(() => {
          this.preview = undefined;
        });
      }
      this.preview.title = path.basename(uri.fsPath);
      this.preview.webview.html = openDoc.getText();
      this.preview.reveal(vscode.ViewColumn.Beside, true);
    }
  }

  private watch(uri: vscode.Uri): void {
    this.watcher?.dispose();
    this.watcher = vscode.workspace.createFileSystemWatcher(uri.fsPath);
    const refresh = async () => {
      if (!this.current || this.current.toString() !== uri.toString()) return;
      try {
        const doc = await vscode.workspace.openTextDocument(uri);
        if (uri.path.endsWith(".html") || uri.path.endsWith(".htm")) {
          if (this.preview) this.preview.webview.html = doc.getText();
        }
      } catch {
        /* file may have been deleted */
      }
    };
    this.watcher.onDidChange(refresh);
    this.watcher.onDidCreate(refresh);
  }

  private async findPlans(): Promise<vscode.Uri[]> {
    const found = await vscode.workspace.findFiles(PLAN_GLOB, "**/node_modules/**", 50);
    return found.sort((a, b) => a.fsPath.localeCompare(b.fsPath));
  }

  dispose(): void {
    this.watcher?.dispose();
    this.preview?.dispose();
  }
}

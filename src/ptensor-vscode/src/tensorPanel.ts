import * as fs from 'fs';
import * as path from 'path';
import * as vscode from 'vscode';
import { NamedTensorJson } from './readTensor';

/**
 * Hosts the tensor-view webview bundle (built from src/ptensor-view) and feeds
 * it tensors as `TensorJson`. All rendering — tables, images, stats — and all
 * dtype decoding live in that bundle; this class only manages the panel
 * lifecycle and the host<->webview message handshake.
 */
export class TensorPanel {
    // One panel (tab) per title. Viewing a new tensor opens a new tab; viewing
    // one that's already open reuses and refreshes its tab.
    private static panels = new Map<string, TensorPanel>();
    private readonly key: string;
    private readonly panel: vscode.WebviewPanel;
    private readonly disposables: vscode.Disposable[] = [];
    private pending: unknown;
    private ready = false;
    // Re-reads the tensor from its source; undefined for demo (sample) panels.
    private refresh?: () => Promise<NamedTensorJson>;

    /** Opens (or reuses) a tab to view a single tensor. */
    static show(
        context: vscode.ExtensionContext,
        tensor: NamedTensorJson,
        refresh?: () => Promise<NamedTensorJson>
    ) {
        TensorPanel.open(context, `Tensor: ${tensor.name}`, tensorMessage(tensor, !!refresh), refresh);
    }

    /** Opens (or reuses) a tab in demo mode: the built-in sample tensors. */
    static showDemo(context: vscode.ExtensionContext) {
        TensorPanel.open(context, 'ptensor: Sample Tensors', { type: 'demo', ...threshold() });
    }

    private static open(
        context: vscode.ExtensionContext,
        title: string,
        message: unknown,
        refresh?: () => Promise<NamedTensorJson>
    ) {
        const column = vscode.window.activeTextEditor?.viewColumn ?? vscode.ViewColumn.Beside;
        const existing = TensorPanel.panels.get(title);
        if (existing) {
            existing.refresh = refresh;
            existing.panel.reveal(column);
            existing.update(title, message);
            return;
        }
        const panel = vscode.window.createWebviewPanel(
            'ptensor.tensorView',
            title,
            column,
            { enableScripts: true, retainContextWhenHidden: true }
        );
        TensorPanel.panels.set(title, new TensorPanel(context, panel, title, message, refresh));
    }

    private constructor(
        context: vscode.ExtensionContext,
        panel: vscode.WebviewPanel,
        title: string,
        message: unknown,
        refresh?: () => Promise<NamedTensorJson>
    ) {
        this.key = title;
        this.panel = panel;
        this.refresh = refresh;
        // First paint is driven by the init message embedded in the HTML below,
        // so no pending message / handshake is needed to render initially.
        this.pending = undefined;
        this.panel.title = title;
        this.panel.onDidDispose(() => this.dispose(), null, this.disposables);
        this.panel.webview.onDidReceiveMessage(
            (msg: { type?: string }) => {
                if (msg?.type === 'ready') {
                    this.ready = true;
                    this.flush();
                } else if (msg?.type === 'refresh') {
                    void this.doRefresh();
                }
            },
            null,
            this.disposables
        );
        this.panel.webview.html = renderHtml(context, this.panel.webview, message);
    }

    /** Re-reads the tensor and pushes the fresh data to the webview. */
    private async doRefresh() {
        if (!this.refresh) {
            return;
        }
        try {
            const tensor = await this.refresh();
            this.update(`Tensor: ${tensor.name}`, tensorMessage(tensor, true));
        } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            vscode.window.showErrorMessage(`ptensor: ${msg}`);
        }
    }

    private update(title: string, message: unknown) {
        this.panel.title = title;
        this.pending = message;
        if (this.ready) {
            this.flush();
        }
    }

    private flush() {
        if (this.pending === undefined) {
            return;
        }
        const message = this.pending;
        this.pending = undefined;
        this.panel.webview.postMessage(message);
    }

    private dispose() {
        if (TensorPanel.panels.get(this.key) === this) {
            TensorPanel.panels.delete(this.key);
        }
        while (this.disposables.length) {
            this.disposables.pop()?.dispose();
        }
    }
    }
}

/** Current table threshold setting, spread into outgoing messages. */
function threshold(): { tableThreshold: number } {
    return {
        tableThreshold: vscode.workspace
            .getConfiguration('ptensor')
            .get<number>('tableElementThreshold', 256),
    };
}

function tensorMessage(tensor: NamedTensorJson, canRefresh: boolean) {
    return { type: 'tensor', name: tensor.name, tensor: tensor.json, canRefresh, ...threshold() };
}

/**
 * Resolves the built webview bundle. Copied into the extension's `media/` at
 * build time (see `copy:viewer` script) so it ships inside the .vsix; falls
 * back to the sibling ptensor-view build for an un-copied dev checkout.
 */
function webviewBundlePath(context: vscode.ExtensionContext): string {
    const packaged = path.join(context.extensionPath, 'media', 'webview.js');
    if (fs.existsSync(packaged)) {
        return packaged;
    }
    return path.join(
        context.extensionPath,
        '..',
        '..',
        'bindings',
        'typescript',
        'ptensor-view',
        'dist',
        'webview.js'
    );
}

function renderHtml(
    context: vscode.ExtensionContext,
    webview: vscode.Webview,
    initMessage: unknown
): string {
    const bundlePath = webviewBundlePath(context);
    let script: string;
    try {
        script = fs.readFileSync(bundlePath, 'utf8');
    } catch {
        return `<!DOCTYPE html><html><body style="font-family: sans-serif; padding: 16px;">
            <h3>ptensor viewer bundle not found</h3>
            <p>Expected build artifact at:</p>
            <pre>${escapeHtml(bundlePath)}</pre>
            <p>Build it with: <code>npm run build:viewer</code> in src/ptensor-vscode.</p>
        </body></html>`;
    }

    const nonce = makeNonce();
    const csp = [
        `default-src 'none'`,
        `img-src ${webview.cspSource} data:`,
        `style-src ${webview.cspSource} 'unsafe-inline'`,
        `script-src 'nonce-${nonce}'`,
    ].join('; ');

    // Embed the first message so the webview renders on load (no handshake
    // round-trip). `<` is escaped to keep the JSON from closing the script tag.
    const initJson = JSON.stringify(initMessage ?? null).replace(/</g, '\\u003c');

    return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="${csp}">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
<div id="root"></div>
<script nonce="${nonce}">window.__PTENSOR_INIT__ = ${initJson};</script>
<script nonce="${nonce}">${script}</script>
</body>
</html>`;
}

function makeNonce(): string {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let out = '';
    for (let i = 0; i < 32; i++) {
        out += chars[Math.floor(Math.random() * chars.length)];
    }
    return out;
}

function escapeHtml(s: string): string {
    return s
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

// Tensors on their way to a panel, as files.
//
// A panel reads its tensor over `fetch` from a webview URI rather than being
// posted the `TensorJson`: the JSON is hundreds of megabytes of base64 for a
// batched image, and `postMessage` into a webview means a stringify, an IPC hop
// and a parse -- several full copies of it through a channel not built for that.
//
// The file is written under the extension's storage directory, which the panel
// names in its `localResourceRoots`, and deleted as soon as the panel is done
// with it.

import * as vscode from 'vscode';
import type { TensorJson } from 'ptensor-ts';
import { fileNameFor } from './tensorFileName';

const DIRECTORY = 'tensors';

/** Writes one tensor per key and hands back a URI the webview may read. */
export class TensorFiles {
    private readonly root: vscode.Uri;
    private readonly written = new Map<string, vscode.Uri>();
    private swept: Promise<void> | undefined;

    constructor(context: vscode.ExtensionContext) {
        // Workspace storage when there is a workspace, global otherwise: either
        // is the extension's own, and neither is inside the opened project.
        const base = context.storageUri ?? context.globalStorageUri;
        this.root = vscode.Uri.joinPath(base, DIRECTORY);
    }

    /** The directory a panel has to allow itself to read from. */
    get directory(): vscode.Uri {
        return this.root;
    }

    /**
     * Writes `tensor` for `key`, replacing whatever was there, and returns the
     * URI to hand the webview. `key` becomes a filename, so it is hashed rather
     * than trusted: it comes from a tensor's name, which the debuggee chose.
     */
    async write(key: string, tensor: TensorJson): Promise<vscode.Uri> {
        await this.sweepOnce();
        const target = vscode.Uri.joinPath(this.root, `${fileNameFor(key)}.json`);
        await vscode.workspace.fs.writeFile(target, Buffer.from(JSON.stringify(tensor), 'utf8'));
        this.written.set(key, target);
        return target;
    }

    /** Drops the file written for `key`, if any. */
    async discard(key: string): Promise<void> {
        const target = this.written.get(key);
        if (target === undefined) {
            return;
        }
        this.written.delete(key);
        await remove(target);
    }

    /** Drops every file this session wrote. */
    async discardAll(): Promise<void> {
        const targets = [...this.written.values()];
        this.written.clear();
        await Promise.all(targets.map(remove));
    }

    /**
     * Creates the directory and clears what a previous session left behind: a
     * crash or a hard reload has no chance to delete its files, and these are
     * hundreds of megabytes each.
     */
    private sweepOnce(): Promise<void> {
        this.swept ??= (async () => {
            await vscode.workspace.fs.createDirectory(this.root);
            let entries: [string, vscode.FileType][];
            try {
                entries = await vscode.workspace.fs.readDirectory(this.root);
            } catch {
                return;
            }
            await Promise.all(
                entries
                    .filter(([name, kind]) => kind === vscode.FileType.File && name.endsWith('.json'))
                    .map(([name]) => remove(vscode.Uri.joinPath(this.root, name)))
            );
        })();
        return this.swept;
    }
}

async function remove(target: vscode.Uri): Promise<void> {
    try {
        await vscode.workspace.fs.delete(target, { useTrash: false });
    } catch {
        // Already gone, or never written. Nothing to do either way.
    }
}

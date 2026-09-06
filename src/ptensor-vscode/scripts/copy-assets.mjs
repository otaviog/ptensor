// Copies the built tensor-view webview bundle into the extension's media/ so it
// ships inside the .vsix. Run after `bun run build:webview` in
// ../../bindings/typescript/ptensor-view.
import { copyFileSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const extRoot = join(dirname(fileURLToPath(import.meta.url)), '..');
const viewRoot = join(extRoot, '..', '..', 'bindings', 'typescript', 'ptensor-view');
const src = join(viewRoot, 'dist', 'webview.js');
const destDir = join(extRoot, 'media');
const dest = join(destDir, 'webview.js');

if (!existsSync(src)) {
    console.error(`tensor-view bundle not found at ${src}\nRun "bun run build:webview" in bindings/typescript/ptensor-view first.`);
    process.exit(1);
}

mkdirSync(destDir, { recursive: true });
copyFileSync(src, dest);
console.log(`Copied webview bundle -> ${dest}`);

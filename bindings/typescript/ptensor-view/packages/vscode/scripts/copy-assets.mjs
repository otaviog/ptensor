// Fills the extension's media/ with what the .vsix ships but does not own: the
// built tensor-view webview bundle, and the icon, whose one copy in the repo
// lives in docs/. media/ is git-ignored -- everything in it is derived.
//
// Run after `bun run build` in ../view.
import { copyFileSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const extRoot = join(dirname(fileURLToPath(import.meta.url)), '..');
const repoRoot = join(extRoot, '..', '..', '..', '..', '..');
const viewRoot = join(extRoot, '..', 'view');
const destDir = join(extRoot, 'media');

// `icon` in package.json has to name a path inside the extension, so the icon
// is copied rather than referenced where it lives.
const assets = [
    {
        what: 'webview bundle',
        src: join(viewRoot, 'dist', 'webview.js'),
        dest: join(destDir, 'webview.js'),
        missing: 'Run "bun run build:webview" in bindings/typescript/ptensor-view first.',
    },
    {
        what: 'icon',
        src: join(repoRoot, 'docs', 'icon-light.png'),
        dest: join(destDir, 'icon.png'),
        missing: 'The repo icon is missing.',
    },
];

mkdirSync(destDir, { recursive: true });
for (const { what, src, dest, missing } of assets) {
    if (!existsSync(src)) {
        console.error(`${what} not found at ${src}\n${missing}`);
        process.exit(1);
    }
    copyFileSync(src, dest);
    console.log(`Copied ${what} -> ${dest}`);
}

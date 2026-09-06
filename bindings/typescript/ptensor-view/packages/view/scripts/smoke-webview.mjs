// Smoke test for the built webview bundle. Executes dist/webview.js in a JSDOM
// window with an embedded demo init message and asserts the React app mounts.
// Guards against regressions that throw on load and leave a blank panel — most
// notably an unreplaced `process.env.NODE_ENV` (no `process` in a webview).
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { JSDOM, VirtualConsole } from 'jsdom';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const bundle = readFileSync(join(root, 'dist', 'webview.js'), 'utf8');
const init = JSON.stringify({ type: 'demo', tableThreshold: 256 }).replace(/</g, '\\u003c');

const html = `<!DOCTYPE html><html><head></head><body>
<div id="root"></div>
<script>window.__PTENSOR_INIT__ = ${init};</script>
<script>${bundle}</script>
</body></html>`;

const errors = [];
const vc = new VirtualConsole();
vc.on('jsdomError', (e) => errors.push(e.message));

const dom = new JSDOM(html, { runScripts: 'dangerously', virtualConsole: vc, pretendToBeVisual: true });

setTimeout(() => {
    const el = dom.window.document.getElementById('root');
    const text = el?.textContent ?? '';
    const ok = errors.length === 0 && el.childElementCount > 0 && text.includes('scalar');
    if (!ok) {
        console.error('webview smoke FAILED');
        if (errors.length) console.error('  errors:', errors.join('; '));
        console.error(`  root children: ${el?.childElementCount}, has "scalar": ${text.includes('scalar')}`);
        process.exit(1);
    }
    console.log('webview smoke OK — panel mounts the sample browser.');
}, 600);

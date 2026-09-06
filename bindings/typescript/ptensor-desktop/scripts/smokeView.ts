// Headless smoke test of the real webview bundle: builds src/webview the way
// the app build does, evaluates it against a happy-dom DOM, and answers its
// RPC with a fake bun host. Catches what unit tests cannot — a bundling
// mistake (e.g. two React copies, which breaks every hook in TensorViewer) and
// a mismatch between the two sides of the RPC schema.

import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { GlobalRegistrator } from '@happy-dom/global-registrator';
import { bytesToBase64 } from 'ptensor-ts';
import { dedupeReact } from '../src/build/dedupeReact';
import type { ServerInfo, TensorPayload } from '../src/shared/rpc';

// ---- build the bundle the app ships -----------------------------------------

const outdir = await mkdtemp(join(tmpdir(), 'ptensor-view-smoke-'));
const build = await Bun.build({
    entrypoints: ['src/webview/index.tsx'],
    outdir,
    target: 'browser',
    plugins: [dedupeReact],
});
if (!build.success) {
    console.error(build.logs.join('\n'));
    throw new Error('view bundle failed to build');
}

const bundlePath = join(outdir, 'index.js');
const bundle = await Bun.file(bundlePath).text();
const reactCopies = new Set(
    [...bundle.matchAll(/\/\/ (\S*node_modules\/react\/index\.js)/g)].map((m) => m[1])
);
if (reactCopies.size > 1) {
    throw new Error(`bundle contains ${reactCopies.size} React copies: ${[...reactCopies]}`);
}

// ---- fake bun host -----------------------------------------------------------

// A cap of two proves PTENSOR_VIEW_HISTORY reaches the window over RPC.
const serverInfo: ServerInfo = {
    host: '127.0.0.1',
    port: 8791,
    listening: true,
    maxTensorsPerSession: 2,
};

function fakeTensor(name: string): TensorPayload {
    const data = new Float32Array([0, 0.25, 0.5, 1]);
    const bytes = new Uint8Array(data.buffer);
    return {
        sessionId: 'smoke',
        name,
        receivedAt: Date.now(),
        tensor: {
            dtype: 'float32',
            shape: [2, 2],
            stride: [2, 1],
            size_bytes: bytes.byteLength,
            encoding: 'base64',
            blob: bytesToBase64(bytes),
        },
    };
}

const seenRequests: string[] = [];

interface RequestPacket {
    type: 'request';
    id: number;
    method: string;
    params: unknown;
}

function answer(packet: RequestPacket): unknown {
    seenRequests.push(packet.method);
    switch (packet.method) {
        case 'getServerInfo':
            return serverInfo;
        default:
            throw new Error(`unexpected request '${packet.method}'`);
    }
}

GlobalRegistrator.register();
document.body.innerHTML = '<div id="root"></div>';

const host = window as unknown as Record<string, any>;
// Normally provided by Electrobun's preload script.
host.__electrobun = {};
host.__electrobunBunBridge = {
    postMessage(raw: string) {
        const packet = JSON.parse(raw);
        if (packet.type !== 'request') {
            return;
        }
        host.__electrobun.receiveMessageFromBun({
            type: 'response',
            id: packet.id,
            success: true,
            payload: answer(packet),
        });
    },
};

/** Sends one bun -> webview push, as BrowserView.rpc.send does. */
function push(id: string, payload: unknown): void {
    host.__electrobun.receiveMessageFromBun({ type: 'message', id, payload });
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 30));

// ---- run ---------------------------------------------------------------------

await import(bundlePath);
await settle();

const requireText = (needle: string, what: string) => {
    if (!document.body.textContent?.includes(needle)) {
        throw new Error(`${what} (expected '${needle}')`);
    }
};

if (!document.querySelector('.app') || !document.querySelector('.sidebar')) {
    throw new Error('the app shell did not render');
}
if (!document.querySelector('style')?.textContent?.includes('.ptv-root')) {
    throw new Error("ptensor-view's stylesheet was not injected");
}
if (!seenRequests.includes('getServerInfo')) {
    throw new Error('the view never requested getServerInfo');
}
requireText('listening on 127.0.0.1:8791', 'the feed status is missing');
requireText('No tensors yet', 'the empty-feed hint is missing');

// A pushed tensor must reach the panel, decoded by the real fromTensorJson.
push('newTensor', fakeTensor('alpha'));
await settle();
requireText('alpha', 'the pushed tensor was not shown');
if (!document.querySelector('.ptv-root')) {
    throw new Error('TensorViewer did not render');
}
requireText('shape=[2, 2]', "TensorViewer's header did not render");

// Follow mode is on, so a second push must move the panel.
push('newTensor', fakeTensor('beta'));
await settle();
requireText('beta', 'the panel did not follow the newest tensor');
requireText('2 kept / 2 received', 'the session footer did not update');

// A third push passes the reported cap, so the oldest tensor is dropped.
push('newTensor', fakeTensor('gamma'));
await settle();
requireText('2 kept / 3 received', 'the reported history cap was not applied');
if (document.body.textContent?.includes('alpha')) {
    throw new Error('the oldest tensor was kept past the history cap');
}

console.log(
    `smoke: bundle renders with one React copy, RPC answered (${seenRequests.join(', ')}), panel follows pushes`
);
// The view has no pending work left; exit without waiting on RPC timeouts.
process.exit(0);

// Main (bun) process: opens the viewer window, runs the local tensor feed, and
// holds what arrives.
//
// The tensors live here, as the JSON text the producer sent. The window reads
// them over HTTP from the feed server (./feedServer) and gets a metadata push
// per arrival, so nothing the size of a tensor is ever handed across as a
// message. That also means a producer can start before the window: what it
// sends is in the store, and the window picks it up when it connects.

import { randomBytes } from 'node:crypto';
import Electrobun, { BrowserWindow } from 'electrobun/bun';
import { configureLogging, disposeLogging, getAppLogger } from '../shared/logging';
import { DEFAULT_BUDGET_BYTES, DEFAULT_PORT, TensorStore } from 'ptensor-tlog';
import { TensorServer } from 'ptensor-tlog/bun';
import { FEED_GLOBAL, type FeedEndpoint } from '../shared/feed';
import { FeedServer } from './feedServer';
import { openLogFile } from './logFile';

/** Reads a positive integer from the environment, or falls back. */
function positiveInt(value: string | undefined, fallback: number): number {
    const parsed = Number(value);
    if (value === undefined || !Number.isFinite(parsed) || parsed < 1) {
        return fallback;
    }
    return Math.floor(parsed);
}

const logFile = openLogFile();
configureLogging({
    level: process.env.PTENSOR_VIEW_LOG_LEVEL,
    sinks: logFile === null ? {} : { file: logFile.sink },
});

const log = getAppLogger('app');
const feedLog = getAppLogger('tensor-feed');
const httpLog = getAppLogger('feed-server');
if (logFile === null) {
    log.warn('No writable log file location; logging to the console only.');
} else {
    log.info('Logging to {path}.', { path: logFile.path });
}

const port = positiveInt(process.env.PTENSOR_VIEW_PORT, DEFAULT_PORT);
const hostname = process.env.PTENSOR_VIEW_HOST ?? '127.0.0.1';
// A byte budget, not a tensor count: one batched float32 image is a couple of
// hundred megabytes, so "keep the last 100" is not a thing that can be done.
const budgetBytes = positiveInt(
    process.env.PTENSOR_VIEW_BUDGET_MB,
    DEFAULT_BUDGET_BYTES / (1024 * 1024)
) * 1024 * 1024;
log.info('Holding up to {budgetMb} MiB of tensors.', { budgetMb: budgetBytes / (1024 * 1024) });

// The line cap has to clear the largest tensor a producer will send, base64 of
// zstd and all. Measured: a 30x3x570x1132 float32 tensor is 222 MiB raw, and
// float mantissas barely compress (1.13x at zstd -1), so the line is ~262 MiB
// -- over ptensor-tlog's 256 MiB default, which would close the connection
// rather than show the tensor. 1 GiB by default, and a producer that needs
// more can say so.
const maxLineBytes = positiveInt(process.env.PTENSOR_VIEW_MAX_LINE_MB, 1024) * 1024 * 1024;

const store = new TensorStore({
    budgetBytes,
    logger: {
        info: (message) => feedLog.info(message),
        warn: (message) => feedLog.warn(message),
    },
});

const server = new TensorServer({
    port,
    hostname,
    maxLineBytes,
    onTensor: (payload) => {
        const added = store.add(payload);
        log.info('Tensor {name} for session {sessionId} stored as {id}.', {
            name: added.meta.name,
            sessionId: added.meta.sessionId,
            id: added.meta.id,
        });
        feedServer.announce(added);
    },
    logger: {
        info: (message) => feedLog.info(message),
        warn: (message) => feedLog.warn(message),
        error: (message) => feedLog.error(message),
    },
});

// Started before the window: the window is told where to connect, so the
// address has to exist first.
const feedServer = new FeedServer({
    store,
    token: randomBytes(32).toString('hex'),
    feedInfo: () => server.serverInfo(),
    logger: httpLog,
});
const address = feedServer.start();
const endpoint: FeedEndpoint = { origin: address.origin, token: feedServer.token };

// With a Vite dev server up (`bun run dev/ui`), the window loads the panel from
// it and picks up UI edits without restarting this process, so the feed and the
// tensors already received survive. Unset, the window loads the bundle
// electrobun built.
const devUrl = process.env.PTENSOR_VIEW_DEV_URL;
if (devUrl !== undefined) {
    log.info('Loading the window from the dev server at {devUrl}.', { devUrl });
}

/**
 * The endpoint reaches the window through the preload script, which runs before
 * the page's own scripts.
 *
 * The URL query is a fallback only for an http(s) window -- the Vite dev
 * server. `views://` is served by a scheme handler that resolves the URL to a
 * bundled file: given a query it looks for `index.html?feed=...`, finds
 * nothing, and the window loads an empty response instead of the panel. So the
 * bundled app gets a bare URL and the preload is the only channel.
 */
function windowUrl(base: string): string {
    if (!base.startsWith('http://') && !base.startsWith('https://')) {
        return base;
    }
    const url = new URL(base);
    url.searchParams.set('feed', endpoint.origin);
    url.searchParams.set('token', endpoint.token);
    return url.toString();
}

const mainWindow = new BrowserWindow({
    title: 'ptensor View',
    url: windowUrl(devUrl ?? 'views://webview/index.html'),
    frame: { x: 120, y: 120, width: 1280, height: 860 },
    preload: `window.${FEED_GLOBAL} = ${JSON.stringify(endpoint)};`,
});

mainWindow.webview.on('dom-ready', () => {
    log.info('Window is up; it reads the feed from {origin}.', { origin: endpoint.origin });
});

Electrobun.events.on('before-quit', () => {
    server.stop();
    feedServer.stop();
    // Flushes and closes the log file.
    disposeLogging();
});

server.start();

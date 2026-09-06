// Main (bun) process: opens the viewer window, runs the local tensor feed, and
// pushes every accepted tensor to the webview. Nothing is stored here -- the
// webview keeps the history it shows.

import Electrobun, { BrowserView, BrowserWindow } from 'electrobun/bun';
import { configureLogging, disposeLogging, getAppLogger } from '../shared/logging';
import { DEFAULT_PORT } from 'ptensor-tlog';
import { TensorServer } from 'ptensor-tlog/bun';
import type { TensorPayload, ViewerRPC } from '../shared/rpc';
import { openLogFile } from './logFile';

/**
 * Tensors that arrive before the webview is up wait here. Capped so a producer
 * that starts before the window cannot grow it without bound.
 */
const PENDING_LIMIT = 32;

/** Tensors the window keeps per session when PTENSOR_VIEW_HISTORY says nothing. */
const DEFAULT_HISTORY = 100;

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
if (logFile === null) {
    log.warn('No writable log file location; logging to the console only.');
} else {
    log.info('Logging to {path}.', { path: logFile.path });
}

const port = positiveInt(process.env.PTENSOR_VIEW_PORT, DEFAULT_PORT);
const hostname = process.env.PTENSOR_VIEW_HOST ?? '127.0.0.1';
// The window keeps the history, so this setting has to travel to it over RPC.
const maxTensorsPerSession = positiveInt(process.env.PTENSOR_VIEW_HISTORY, DEFAULT_HISTORY);
log.info('Keeping {maxTensorsPerSession} tensors per session.', { maxTensorsPerSession });

const rpc = BrowserView.defineRPC<ViewerRPC>({
    handlers: {
        requests: {
            getServerInfo: () => ({ ...server.serverInfo(), maxTensorsPerSession }),
        },
        messages: {},
    },
});

let ready = false;
const pending: TensorPayload[] = [];

function pushTensor(payload: TensorPayload): void {
    log.info('Tensor {name} for session {sessionId} ({state}).', {
        name: payload.name,
        sessionId: payload.sessionId,
        state: ready ? 'pushed to the window' : 'held until the window is up',
    });
    if (!ready) {
        pending.push(payload);
        if (pending.length > PENDING_LIMIT) {
            pending.splice(0, pending.length - PENDING_LIMIT);
        }
        return;
    }
    mainWindow.webview.rpc?.send.newTensor(payload);
}

const server = new TensorServer({
    port,
    hostname,
    onTensor: pushTensor,
    logger: {
        info: (message) => feedLog.info(message),
        warn: (message) => feedLog.warn(message),
        error: (message) => feedLog.error(message),
    },
});

// With a Vite dev server up (`bun run dev:ui`), the window loads the panel from
// it and picks up UI edits without restarting this process, so the feed and the
// tensors it already received survive. Unset, the window loads the bundle
// electrobun built.
const devUrl = process.env.PTENSOR_VIEW_DEV_URL;
if (devUrl !== undefined) {
    log.info('Loading the window from the dev server at {devUrl}.', { devUrl });
}

const mainWindow = new BrowserWindow({
    title: 'ptensor View',
    url: devUrl ?? 'views://webview/index.html',
    frame: { x: 120, y: 120, width: 1280, height: 860 },
    rpc,
});

mainWindow.webview.on('dom-ready', () => {
    ready = true;
    log.info('Window is up, flushing {pending} held tensors.', { pending: pending.length });
    for (const payload of pending.splice(0)) {
        mainWindow.webview.rpc?.send.newTensor(payload);
    }
});

Electrobun.events.on('before-quit', () => {
    server.stop();
    // Flushes and closes the log file.
    disposeLogging();
});

server.start();

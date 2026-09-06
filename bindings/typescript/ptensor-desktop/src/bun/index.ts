// Main (bun) process: opens the viewer window, runs the local tensor feed, and
// pushes every accepted tensor to the webview. Nothing is stored here -- the
// webview keeps the history it shows.

import Electrobun, { BrowserView, BrowserWindow } from 'electrobun/bun';
import { configureLogging, disposeLogging, getAppLogger } from '../shared/logging';
import { DEFAULT_PORT } from './server/constants';
import type { TensorPayload, ViewerRPC } from '../shared/rpc';
import { TensorServer } from './server/tensorServer';
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
});

const mainWindow = new BrowserWindow({
    title: 'ptensor View',
    url: 'views://webview/index.html',
    frame: { x: 120, y: 120, width: 1280, height: 860 },
    rpc,
});

mainWindow.webview.on('dom-ready', () => {
    ready = true;
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

// Main (bun) process: opens the viewer window, runs the local tensor feed, and
// serves the webview's requests against the in-memory session store.

import Electrobun, { BrowserView, BrowserWindow } from 'electrobun/bun';
import { configureLogging, disposeLogging, getAppLogger } from '../shared/logging';
import { DEFAULT_PORT } from '../shared/protocol';
import type { ViewerRPC } from '../shared/rpc';
import { openLogFile } from './logFile';
import { SessionStore } from './sessionStore';
import { TensorServer } from './tensorServer';

/** Pushes are coalesced over this window so a fast producer cannot flood RPC. */
const PUSH_INTERVAL_MS = 100;

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

const port = Number(process.env.PTENSOR_VIEW_PORT ?? DEFAULT_PORT);
const hostname = process.env.PTENSOR_VIEW_HOST ?? '127.0.0.1';
const maxTensorsPerSession = Number(process.env.PTENSOR_VIEW_HISTORY ?? 100);

const store = new SessionStore({ maxTensorsPerSession });

const rpc = BrowserView.defineRPC<ViewerRPC>({
    handlers: {
        requests: {
            getSessions: () => store.summaries(),
            getServerInfo: () => server.serverInfo(),
            getTensor: ({ sessionId, tensorId }) => store.payload(sessionId, tensorId),
            clearSession: ({ sessionId }) => {
                store.clear(sessionId);
                pushSessions();
            },
        },
        messages: {},
    },
});

let pushTimer: ReturnType<typeof setTimeout> | null = null;

/** Sends the current summaries, at most once per `PUSH_INTERVAL_MS`. */
function schedulePush(): void {
    if (pushTimer !== null) {
        return;
    }
    pushTimer = setTimeout(() => {
        pushTimer = null;
        pushSessions();
    }, PUSH_INTERVAL_MS);
}

function pushSessions(): void {
    mainWindow.webview.rpc?.send.sessions(store.summaries());
}

const server = new TensorServer(store, {
    port,
    hostname,
    onChange: schedulePush,
});

const mainWindow = new BrowserWindow({
    title: 'ptensor View',
    url: 'views://mainview/index.html',
    frame: { x: 120, y: 120, width: 1280, height: 860 },
    rpc,
});

mainWindow.webview.on('dom-ready', () => {
    mainWindow.webview.rpc?.send.serverInfo(server.serverInfo());
    pushSessions();
});

Electrobun.events.on('before-quit', () => {
    server.stop();
    // Flushes and closes the log file.
    disposeLogging();
});

server.start();

// Webview entry: mounts the React app and bridges it to the bun process over
// Electrobun RPC. Stylesheets are imported as text and injected, so the view
// ships as one JS bundle (ptensor-view's CSS lives in its own package).

import Electrobun, { Electroview } from 'electrobun/view';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { fromTensorJson, type TensorView } from '@ptensor/tensor-view';
import tensorViewCss from '@ptensor/tensor-view/styles.css' with { type: 'text' };
import appCss from './app.css' with { type: 'text' };
import type { ServerInfo, SessionSummary } from '../shared/protocol';
import { getAppLogger } from '../shared/logging';
import type { ViewerRPC } from '../shared/rpc';
import { App } from './App';

const log = getAppLogger('mainview');

let onSessions: (sessions: SessionSummary[]) => void = () => {};
let onServerInfo: (info: ServerInfo) => void = () => {};

const rpc = Electroview.defineRPC<ViewerRPC>({
    handlers: {
        requests: {},
        messages: {
            sessions: (payload) => onSessions(payload),
            serverInfo: (payload) => onServerInfo(payload),
        },
    },
});

const electrobun = new Electrobun.Electroview({ rpc });

function injectStyles(): void {
    const style = document.createElement('style');
    style.textContent = `${tensorViewCss}\n${appCss}`;
    document.head.appendChild(style);
}

async function loadTensor(sessionId: string, tensorId: string): Promise<TensorView | null> {
    const payload = await electrobun.rpc?.request.getTensor({ sessionId, tensorId });
    return payload ? fromTensorJson(payload.tensor, payload.name) : null;
}

function clearSession(sessionId?: string): void {
    electrobun.rpc?.request.clearSession({ sessionId }).catch(reportRpcError);
}

/** An unanswered request should not surface as an unhandled rejection. */
function reportRpcError(error: unknown): void {
    log.error('RPC request failed: {error}', { error });
}

/**
 * Wires the pushed-message handlers, then pulls the current state once so a
 * reloaded webview does not sit empty until the next tensor arrives.
 */
function subscribe(
    sessions: (next: SessionSummary[]) => void,
    info: (next: ServerInfo) => void
): void {
    onSessions = sessions;
    onServerInfo = info;
    electrobun.rpc?.request.getSessions({}).then(sessions).catch(reportRpcError);
    electrobun.rpc?.request.getServerInfo({}).then(info).catch(reportRpcError);
}

injectStyles();
createRoot(document.getElementById('root') as HTMLElement).render(
    <StrictMode>
        <App loadTensor={loadTensor} clearSession={clearSession} subscribe={subscribe} />
    </StrictMode>
);

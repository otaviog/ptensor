// Webview entry: mounts the React app and bridges it to the bun process over
// Electrobun RPC. Stylesheets are imported as text and injected, so the view
// ships as one JS bundle (ptensor-view's CSS lives in its own package).

import Electrobun, { Electroview } from 'electrobun/view';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import appCss from './app.css' with { type: 'text' };
import tensorViewCss from '@ptensor/tensor-view/styles.css' with { type: 'text' };
import { getAppLogger } from '../shared/logging';
import type { ServerInfo, TensorPayload, ViewerRPC } from '../shared/rpc';
import { App } from './App';

const log = getAppLogger('webview');

// Pushes that land before App has subscribed are held here: the bun process
// flushes whatever arrived while the window was coming up.
const early: TensorPayload[] = [];
let onTensor: (payload: TensorPayload) => void = (payload) => early.push(payload);

const rpc = Electroview.defineRPC<ViewerRPC>({
    handlers: {
        requests: {},
        messages: {
            newTensor: (payload) => onTensor(payload),
        },
    },
});

const electrobun = new Electrobun.Electroview({ rpc });

function injectStyles(): void {
    const style = document.createElement('style');
    style.textContent = `${tensorViewCss}\n${appCss}`;
    document.head.appendChild(style);
}

function subscribe(next: (payload: TensorPayload) => void): void {
    onTensor = next;
    for (const payload of early.splice(0)) {
        next(payload);
    }
}

async function getServerInfo(): Promise<ServerInfo> {
    const info = await electrobun.rpc?.request.getServerInfo({});
    if (!info) {
        throw new Error('the bun process did not answer getServerInfo');
    }
    return info;
}

injectStyles();
createRoot(document.getElementById('root') as HTMLElement).render(
    <StrictMode>
        <App subscribe={subscribe} getServerInfo={getServerInfo} />
    </StrictMode>
);

log.debug('Renderer mounted.');

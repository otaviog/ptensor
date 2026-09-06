// Bridges the React app to the bun process over Electrobun RPC and mounts it.
//
// Styles are left to the caller: the production entry (./index.tsx) injects
// them as text, the Vite dev entry (./dev/main.tsx) imports them so the dev
// server can hot-replace them. Everything else is shared, so what runs under
// `vite` is what ships.

import Electrobun, { Electroview } from 'electrobun/view';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
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

/** Renders the panel into `#root`. Call once, after the styles are in place. */
export function mount(): void {
    createRoot(document.getElementById('root') as HTMLElement).render(
        <StrictMode>
            <App subscribe={subscribe} getServerInfo={getServerInfo} />
        </StrictMode>
    );
    log.debug('Renderer mounted.');
}

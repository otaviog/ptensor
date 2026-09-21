// Connects the React app to the feed the bun process serves and mounts it.
//
// Styles are left to the caller: the production entry (./index.tsx) injects
// them as text, the Vite dev entry (./dev/main.tsx) imports them so the dev
// server can hot-replace them. Everything else is shared, so what runs under
// `vite` is what ships.

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { getAppLogger } from '../shared/logging';
import { App } from './App';
import { createFeedClient, readEndpoint } from './feedClient';

const log = getAppLogger('webview');

/** Renders the panel into `#root`. Call once, after the styles are in place. */
export function mount(): void {
    const root = createRoot(document.getElementById('root') as HTMLElement);
    const endpoint = readEndpoint();

    if (endpoint === null) {
        // Nothing to connect to, so say that rather than render an empty panel
        // that looks like a feed with no tensors on it.
        log.error('No feed endpoint: neither the preload nor the window URL carried one.');
        root.render(
            <div className="placeholder">
                This window was opened without a feed address, so there is nothing to read.
            </div>
        );
        return;
    }

    log.debug('Reading the feed from {origin}.', { origin: endpoint.origin });
    root.render(
        <StrictMode>
            <App client={createFeedClient(endpoint)} />
        </StrictMode>
    );
    log.debug('Renderer mounted.');
}

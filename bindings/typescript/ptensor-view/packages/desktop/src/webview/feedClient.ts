// The window's side of the feed: a socket for the metadata pushes and a fetch
// per tensor the user actually looks at.
//
// A tensor is decoded here, in the window, and only here -- the bun process
// hands over the `TensorJson` it received and never touches the blob. The
// decode itself goes through ptensor-ts, which picks the engine's native base64
// where there is one.

import { tensorFromJson, type Tensor, type TensorJson } from '@ptensor/tensor-view';
import {
    FEED_GLOBAL,
    FEED_ROUTES,
    type FeedEndpoint,
    type FeedEvent,
    type TensorMeta,
} from '../shared/feed';
import { getAppLogger } from '../shared/logging';

const log = getAppLogger('feed-client');

/** How long a dropped socket waits before dialling again. */
const RECONNECT_MS = 1000;

/** What App talks to. Small on purpose, so a test can stand one up by hand. */
export interface FeedClient {
    /** Starts listening. Returns the unsubscribe. */
    subscribe(onEvent: (event: FeedEvent) => void): () => void;
    /** Every held tensor's metadata, newest first. */
    listTensors(): Promise<TensorMeta[]>;
    /** Fetches one tensor and decodes it. */
    loadTensor(id: string): Promise<Tensor>;
    /** Drops one session's tensors, or all of them. */
    clear(sessionId?: string): Promise<void>;
}

/**
 * Where the bun process said the feed is. It arrives on `window` from the
 * preload script, and on the URL as a fallback -- the preload runs before the
 * page either way, but a preload that quietly did not run would leave the panel
 * with no feed and no way to say so.
 */
export function readEndpoint(): FeedEndpoint | null {
    const injected = (window as unknown as Record<string, unknown>)[FEED_GLOBAL];
    if (isEndpoint(injected)) {
        return injected;
    }
    const params = new URLSearchParams(window.location.search);
    const origin = params.get('feed');
    const token = params.get('token');
    if (origin !== null && token !== null) {
        log.debug('Feed endpoint came from the window URL, not the preload.');
        return { origin, token };
    }
    return null;
}

function isEndpoint(value: unknown): value is FeedEndpoint {
    if (typeof value !== 'object' || value === null) {
        return false;
    }
    const candidate = value as Record<string, unknown>;
    return typeof candidate.origin === 'string' && typeof candidate.token === 'string';
}

export function createFeedClient(endpoint: FeedEndpoint): FeedClient {
    const url = (path: string, params: Record<string, string> = {}): URL => {
        const target = new URL(path, endpoint.origin);
        target.searchParams.set('token', endpoint.token);
        for (const [key, value] of Object.entries(params)) {
            target.searchParams.set(key, value);
        }
        return target;
    };

    async function json<T>(target: URL): Promise<T> {
        const response = await fetch(target);
        if (!response.ok) {
            throw new Error(`${target.pathname} answered ${response.status}`);
        }
        return (await response.json()) as T;
    }

    return {
        subscribe(onEvent) {
            let socket: WebSocket | null = null;
            let retry: ReturnType<typeof setTimeout> | undefined;
            let closed = false;

            const open = (): void => {
                const target = url(FEED_ROUTES.feed);
                target.protocol = target.protocol === 'https:' ? 'wss:' : 'ws:';
                socket = new WebSocket(target.toString());
                socket.addEventListener('message', (event: MessageEvent) => {
                    if (typeof event.data !== 'string') {
                        return;
                    }
                    try {
                        onEvent(JSON.parse(event.data) as FeedEvent);
                    } catch (error: unknown) {
                        log.error('Dropped an unreadable feed event: {error}.', { error });
                    }
                });
                socket.addEventListener('close', () => {
                    if (!closed) {
                        retry = setTimeout(open, RECONNECT_MS);
                    }
                });
            };
            open();

            return () => {
                closed = true;
                clearTimeout(retry);
                socket?.close();
            };
        },

        listTensors() {
            return json<TensorMeta[]>(url(FEED_ROUTES.tensors));
        },

        async loadTensor(id) {
            const target = url(`${FEED_ROUTES.tensor}/${encodeURIComponent(id)}`);
            const response = await fetch(target);
            if (!response.ok) {
                throw new Error(`tensor ${id} answered ${response.status}`);
            }
            return tensorFromJson((await response.json()) as TensorJson);
        },

        async clear(sessionId) {
            const target = url(
                FEED_ROUTES.tensors,
                sessionId === undefined ? {} : { session: sessionId }
            );
            const response = await fetch(target, { method: 'DELETE' });
            if (!response.ok) {
                throw new Error(`clearing answered ${response.status}`);
            }
        },
    };
}

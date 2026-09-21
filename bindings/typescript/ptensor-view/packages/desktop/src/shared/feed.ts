// Contract between the bun process (src/bun, owns the tlog feed and the store)
// and the webview (src/webview, renders the panel).
//
// Tensors do not travel as messages. The bun process holds each one as the
// `TensorJson` text it arrived as and serves it over HTTP; what crosses as a
// message is only the metadata that tells the panel a tensor exists. A tensor
// is hundreds of megabytes of base64, and every message channel between a host
// process and a webview stringifies -- some of them badly enough to spend
// gigabytes on one frame.

import type { FeedInfo, TensorMeta } from 'ptensor-tlog';

export type { FeedInfo, TensorMeta };

/** Pushed down the `/feed` socket. */
export type FeedEvent =
    /** First message on a fresh connection: the feed's state and what is held. */
    | { type: 'hello'; info: FeedInfo; tensors: TensorMeta[] }
    /**
     * One tensor was accepted. `dropped` names the ids the store had to evict
     * to fit it, so a window listing what is held stays in step with the store
     * instead of finding out on a 404.
     */
    | { type: 'tensor'; tensor: TensorMeta; dropped: string[] }
    /** Tensors were dropped: one session's, or every one of them. */
    | { type: 'cleared'; sessionId?: string };

/**
 * Where the webview finds the feed, and the secret that lets it in. The bun
 * process picks a port at startup, so this cannot be a constant -- it reaches
 * the window through a preload script, with the window URL's query as a
 * fallback (see src/bun/index.ts).
 */
export interface FeedEndpoint {
    /** e.g. `http://127.0.0.1:51234`. */
    origin: string;
    token: string;
}

/** The name the endpoint is injected under. */
export const FEED_GLOBAL = '__ptensorFeed';

export const FEED_ROUTES = {
    /** GET: every held tensor's metadata, newest first. DELETE: drop them. */
    tensors: '/tensors',
    /** GET `/tensor/<id>`: that tensor's `TensorJson`, exactly as it arrived. */
    tensor: '/tensor',
    /** WebSocket: `FeedEvent`s. */
    feed: '/feed',
} as const;

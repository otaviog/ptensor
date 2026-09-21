// The HTTP + WebSocket server the window reads the feed from.
//
// Two jobs, split by size. The socket pushes metadata -- a few hundred bytes
// saying a tensor arrived -- and `GET /tensor/<id>` hands over the tensor
// itself, as the stored JSON text, with no decode and no per-request work.
//
// It listens on loopback with an OS-picked port and every request carries a
// token minted at startup, so another local process cannot read the tensors of
// whatever the user happens to be debugging.

import { timingSafeEqual } from 'node:crypto';
import type { AddResult, FeedInfo, TensorStore } from 'ptensor-tlog';
import { FEED_ROUTES, type FeedEvent } from '../shared/feed';

export interface FeedServerLogger {
    info?(message: string, properties?: Record<string, unknown>): void
    warn?(message: string, properties?: Record<string, unknown>): void
    error?(message: string, properties?: Record<string, unknown>): void
}

export interface FeedServerOptions {
    store: TensorStore;
    /** Secret every request has to carry, as `?token=`. */
    token: string;
    /** State of the tlog feed the tensors come from, read per connection. */
    feedInfo: () => FeedInfo;
    /** Loopback by default; there is no reason to serve tensors off-host. */
    hostname?: string;
    /** 0, the default, lets the OS pick. */
    port?: number;
    logger?: FeedServerLogger;
}

/** What the window was told to connect to. */
export interface FeedServerAddress {
    origin: string;
    port: number;
}

export class FeedServer {
    private readonly log: FeedServerLogger;
    private readonly hostname: string;
    private server: Bun.Server<undefined> | null = null;
    private readonly clients = new Set<Bun.ServerWebSocket<undefined>>();

    constructor(private readonly options: FeedServerOptions) {
        this.log = options.logger ?? {};
        this.hostname = options.hostname ?? '127.0.0.1';
    }

    /** The secret the window has to send. Handed to it with the address. */
    get token(): string {
        return this.options.token;
    }

    /** Binds the port. Throws if it cannot: the window has nothing to read. */
    start(): FeedServerAddress {
        const self = this;
        this.server = Bun.serve({
            hostname: this.hostname,
            port: this.options.port ?? 0,
            fetch: (request, server) => self.route(request, server),
            websocket: {
                open: (ws) => self.onOpen(ws),
                close: (ws) => {
                    self.clients.delete(ws);
                },
                // The window only listens; nothing it could say is acted on.
                message: () => {},
            },
        });
        const port = this.server.port;
        if (port === undefined) {
            throw new Error('the feed server bound no TCP port');
        }
        const address = { origin: `http://${this.hostname}:${port}`, port };
        this.log.info?.('Serving the feed on {origin}.', address);
        return address;
    }

    stop(): void {
        this.server?.stop(true);
        this.server = null;
        this.clients.clear();
    }

    /** Tells every open window that a tensor arrived, and what it displaced. */
    announce(added: AddResult): void {
        this.broadcast({
            type: 'tensor',
            tensor: added.meta,
            dropped: added.evicted.map((meta) => meta.id),
        });
    }

    private broadcast(event: FeedEvent): void {
        const message = JSON.stringify(event);
        for (const client of this.clients) {
            client.send(message);
        }
    }

    private onOpen(ws: Bun.ServerWebSocket<undefined>): void {
        this.clients.add(ws);
        // Whatever arrived before this window came up is already in the store,
        // so a fresh connection is caught up by the hello, not by a replay.
        const hello: FeedEvent = {
            type: 'hello',
            info: this.options.feedInfo(),
            tensors: this.options.store.list(),
        };
        ws.send(JSON.stringify(hello));
    }

    private route(request: Request, server: Bun.Server<undefined>): Response | undefined {
        const url = new URL(request.url);

        if (!this.authorized(url)) {
            return withCors(new Response('forbidden', { status: 403 }));
        }
        if (request.method === 'OPTIONS') {
            return withCors(new Response(null, { status: 204 }));
        }
        if (url.pathname === FEED_ROUTES.feed) {
            // Returning nothing hands the connection to the websocket handlers.
            return server.upgrade(request)
                ? undefined
                : withCors(new Response('expected a websocket upgrade', { status: 400 }));
        }
        if (url.pathname === FEED_ROUTES.tensors) {
            return this.tensors(request, url);
        }
        if (url.pathname.startsWith(`${FEED_ROUTES.tensor}/`)) {
            return this.tensor(url.pathname.slice(FEED_ROUTES.tensor.length + 1));
        }
        return withCors(new Response('not found', { status: 404 }));
    }

    private tensors(request: Request, url: URL): Response {
        if (request.method === 'DELETE') {
            const sessionId = url.searchParams.get('session') ?? undefined;
            this.options.store.clear(sessionId);
            this.broadcast({ type: 'cleared', sessionId });
            return withCors(new Response(null, { status: 204 }));
        }
        return withCors(Response.json(this.options.store.list()));
    }

    private tensor(id: string): Response {
        // Held as text, so this is the string as the producer sent it -- no
        // re-encoding, and nothing here ever decodes a blob.
        const json = this.options.store.json(decodeURIComponent(id));
        if (json === undefined) {
            return withCors(new Response('no such tensor', { status: 404 }));
        }
        return withCors(
            new Response(json, { headers: { 'content-type': 'application/json' } })
        );
    }

    private authorized(url: URL): boolean {
        return constantTimeEqual(url.searchParams.get('token') ?? '', this.options.token);
    }
}

/**
 * The window may be served from the Vite dev server rather than `views://`, so
 * its requests are cross-origin. The token is the guard, not the origin, and no
 * credentials are involved.
 */
function withCors(response: Response): Response {
    response.headers.set('access-control-allow-origin', '*');
    response.headers.set('access-control-allow-methods', 'GET, DELETE, OPTIONS');
    return response;
}

function constantTimeEqual(a: string, b: string): boolean {
    const left = Buffer.from(a);
    const right = Buffer.from(b);
    // timingSafeEqual throws on a length mismatch, which is itself a leak of
    // the length -- but the token is a fixed-width hex string, so a wrong
    // length is a wrong token and there is nothing to learn from it.
    return left.length === right.length && timingSafeEqual(left, right);
}

// End to end over a real socket: the feed server is plain HTTP and WebSocket,
// so it is testable without electrobun and without a window.

import { afterEach, describe, expect, test } from 'bun:test';
import { TensorStore } from 'ptensor-tlog';
import type { FeedInfo, TensorPayload } from 'ptensor-tlog';
import type { TensorJson } from 'ptensor-ts';
import type { FeedEvent } from '../../shared/feed';
import { FeedServer } from '../feedServer';

const TOKEN = 'a'.repeat(64);
const INFO: FeedInfo = { host: '127.0.0.1', port: 4449, listening: true };

function tensor(): TensorJson {
    return {
        dtype: 'float32',
        shape: [2, 2],
        stride: [2, 1],
        size_bytes: 16,
        encoding: 'base64',
        blob: 'AAAAAAAAAAAAAAAAAAAAAA==',
    };
}

function payload(name: string, sessionId = 'run-a'): TensorPayload {
    return { sessionId, name, tensor: tensor(), receivedAt: Date.now() };
}

let running: FeedServer | null = null;

afterEach(() => {
    running?.stop();
    running = null;
});

function start(store: TensorStore) {
    const server = new FeedServer({ store, token: TOKEN, feedInfo: () => INFO });
    running = server;
    const address = server.start();
    return {
        server,
        url: (path: string, token = TOKEN) =>
            `${address.origin}${path}${path.includes('?') ? '&' : '?'}token=${token}`,
        wsUrl: (token = TOKEN) =>
            `${address.origin.replace('http:', 'ws:')}/feed?token=${token}`,
    };
}

/** Collects feed events until `count` of them have arrived. */
function collect(url: string, count: number): Promise<{ events: FeedEvent[]; socket: WebSocket }> {
    return new Promise((resolve, reject) => {
        const socket = new WebSocket(url);
        const events: FeedEvent[] = [];
        const timer = setTimeout(() => reject(new Error('timed out waiting for feed events')), 2000);
        socket.addEventListener('message', (event) => {
            events.push(JSON.parse(String(event.data)) as FeedEvent);
            if (events.length === count) {
                clearTimeout(timer);
                resolve({ events, socket });
            }
        });
        socket.addEventListener('error', (event) => {
            clearTimeout(timer);
            reject(new Error(`socket failed: ${String(event)}`));
        });
    });
}

describe('FeedServer', () => {
    test('turns away a request with no token, or the wrong one', async () => {
        const store = new TensorStore();
        const feed = start(store);

        const origin = feed.url('/tensors').split('?')[0];
        expect((await fetch(origin)).status).toBe(403);
        expect((await fetch(feed.url('/tensors', 'nope'))).status).toBe(403);
        // A token of the right length but the wrong value is still no.
        expect((await fetch(feed.url('/tensors', 'b'.repeat(64)))).status).toBe(403);
        expect((await fetch(feed.url('/tensors'))).status).toBe(200);
    });

    test('lists what is held, newest first, without any blob', async () => {
        const store = new TensorStore();
        const feed = start(store);
        store.add(payload('first'));
        store.add(payload('second'));

        const listed = (await (await fetch(feed.url('/tensors'))).json()) as unknown[];

        expect(listed.map((meta) => (meta as { name: string }).name)).toEqual([
            'second',
            'first',
        ]);
        expect(JSON.stringify(listed)).not.toContain('blob');
    });

    test('serves a tensor as the exact JSON text it arrived as', async () => {
        const store = new TensorStore();
        const feed = start(store);
        const entry = payload('frame');
        const { meta } = store.add(entry);

        const response = await fetch(feed.url(`/tensor/${meta.id}`));

        expect(response.status).toBe(200);
        expect(response.headers.get('content-type')).toContain('application/json');
        // Byte for byte what the producer sent: nothing here re-encodes a blob.
        expect(await response.text()).toBe(JSON.stringify(entry.tensor));
    });

    test('answers 404 for a tensor it does not hold', async () => {
        const feed = start(new TensorStore());

        expect((await fetch(feed.url('/tensor/nothing'))).status).toBe(404);
    });

    test('clears a session on request, and everything without one', async () => {
        const store = new TensorStore();
        const feed = start(store);
        store.add(payload('first', 'run-a'));
        store.add(payload('other', 'run-b'));

        expect((await fetch(feed.url('/tensors?session=run-a'), { method: 'DELETE' })).status).toBe(
            204
        );
        expect(store.list().map((meta) => meta.sessionId)).toEqual(['run-b']);

        await fetch(feed.url('/tensors'), { method: 'DELETE' });
        expect(store.list()).toEqual([]);
    });

    test('greets a new socket with the feed state and what is held', async () => {
        const store = new TensorStore();
        const feed = start(store);
        store.add(payload('first'));

        const { events, socket } = await collect(feed.wsUrl(), 1);
        socket.close();

        expect(events[0]).toEqual({
            type: 'hello',
            info: INFO,
            tensors: store.list(),
        });
    });

    test('announces an arrival, and what it displaced', async () => {
        const store = new TensorStore({ budgetBytes: 1 });
        const feed = start(store);
        const { meta: first } = store.add(payload('first'));

        const waiting = collect(feed.wsUrl(), 2);
        // Let the hello go out before the announcement.
        await Bun.sleep(20);
        feed.server.announce(store.add(payload('second')));

        const { events, socket } = await waiting;
        socket.close();

        expect(events[1]).toMatchObject({
            type: 'tensor',
            tensor: { name: 'second' },
            dropped: [first.id],
        });
    });
});

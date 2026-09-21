// The client against the real server: a live socket, real HTTP, and a real
// base64+zstd blob decoded at the end of it. What this covers that the App
// tests cannot is the wiring itself -- the URLs, the token, the ws:// swap and
// the decode -- none of which a fake client would exercise.

import { afterEach, describe, expect, test } from 'bun:test';
import { TensorStore } from 'ptensor-tlog';
import type { FeedInfo, TensorPayload } from 'ptensor-tlog';
import type { TensorJson } from 'ptensor-ts';
import { FeedServer } from '../../bun/feedServer';
import type { FeedEvent } from '../../shared/feed';
import { createFeedClient, readEndpoint } from '../feedClient';
import { FEED_GLOBAL } from '../../shared/feed';

const TOKEN = 'c'.repeat(64);
const INFO: FeedInfo = { host: '127.0.0.1', port: 4449, listening: true };

/** A tensor on the wire the way `p10::tlog` sends one: base64 over zstd. */
function tensorJson(values: number[]): TensorJson {
    const data = new Float32Array(values);
    const bytes = new Uint8Array(data.buffer);
    return {
        dtype: 'float32',
        shape: [values.length],
        stride: [1],
        size_bytes: bytes.byteLength,
        encoding: 'base64+zstd',
        blob: Buffer.from(Bun.zstdCompressSync(bytes)).toString('base64'),
    };
}

function payload(name: string, values: number[], sessionId = 'run-a'): TensorPayload {
    return { sessionId, name, tensor: tensorJson(values), receivedAt: Date.now() };
}

let running: FeedServer | null = null;

afterEach(() => {
    running?.stop();
    running = null;
    delete (globalThis as unknown as Record<string, unknown>)[FEED_GLOBAL];
});

function start(store: TensorStore) {
    const server = new FeedServer({ store, token: TOKEN, feedInfo: () => INFO });
    running = server;
    const address = server.start();
    return { server, client: createFeedClient({ origin: address.origin, token: TOKEN }) };
}

describe('readEndpoint', () => {
    test('takes the endpoint the preload injected', () => {
        (globalThis as unknown as Record<string, unknown>)[FEED_GLOBAL] = {
            origin: 'http://127.0.0.1:1234',
            token: 'abc',
        };

        expect(readEndpoint()).toEqual({ origin: 'http://127.0.0.1:1234', token: 'abc' });
    });

    test('rejects an injected value that is not an endpoint', () => {
        (globalThis as unknown as Record<string, unknown>)[FEED_GLOBAL] = { origin: 42 };

        // Falls through to the URL, which carries nothing in the test DOM.
        expect(readEndpoint()).toBeNull();
    });
});

describe('createFeedClient', () => {
    test('lists what the store holds', async () => {
        const store = new TensorStore();
        const { client } = start(store);
        store.add(payload('first', [1, 2]));
        store.add(payload('second', [3, 4]));

        const listed = await client.listTensors();

        expect(listed.map((meta) => meta.name)).toEqual(['second', 'first']);
    });

    test('fetches a tensor and decodes its blob', async () => {
        const store = new TensorStore();
        const { client } = start(store);
        const { meta } = store.add(payload('frame', [1, -2, 0.5, 4]));

        const tensor = await client.loadTensor(meta.id);

        expect(tensor.dtype).toBe('float32');
        expect(tensor.shape).toEqual([4]);
        expect([...tensor.data]).toEqual([1, -2, 0.5, 4]);
    });

    test('reports a tensor the store no longer holds', async () => {
        const { client } = start(new TensorStore());

        await expect(client.loadTensor('gone')).rejects.toThrow(/404/);
    });

    test('receives the greeting and then each arrival over the socket', async () => {
        const store = new TensorStore();
        const { server, client } = start(store);
        store.add(payload('first', [1]));

        const events: FeedEvent[] = [];
        const unsubscribe = client.subscribe((event) => events.push(event));
        await waitFor(() => events.length === 1);

        expect(events[0]).toMatchObject({ type: 'hello', info: INFO });
        expect((events[0] as { tensors: unknown[] }).tensors).toHaveLength(1);

        server.announce(store.add(payload('second', [2])));
        await waitFor(() => events.length === 2);

        expect(events[1]).toMatchObject({ type: 'tensor', tensor: { name: 'second' } });

        unsubscribe();
    });

    test('asks the store to clear, by session and entirely', async () => {
        const store = new TensorStore();
        const { client } = start(store);
        store.add(payload('first', [1], 'run-a'));
        store.add(payload('other', [2], 'run-b'));

        await client.clear('run-a');
        expect(store.list().map((meta) => meta.sessionId)).toEqual(['run-b']);

        await client.clear();
        expect(store.list()).toEqual([]);
    });
});

/** Polls until `done`, so a test does not sleep for a fixed guess. */
async function waitFor(done: () => boolean, timeoutMs = 2000): Promise<void> {
    const deadline = Date.now() + timeoutMs;
    while (!done()) {
        if (Date.now() > deadline) {
            throw new Error('timed out');
        }
        await Bun.sleep(5);
    }
}

import { describe, expect, test } from 'bun:test';
import { bytesToBase64 } from 'ptensor-ts';
import { TensorServer } from '../tensorServer';
import { SessionStore } from '../sessionStore';
import type { TensorMessage } from '../../shared/protocol';

function tensorMessage(session: string, name: string, values: number[]): TensorMessage {
    const data = new Float32Array(values);
    return {
        type: 'tensor',
        session,
        name,
        tensor: {
            dtype: 'float32',
            shape: [values.length],
            stride: [1],
            blob: bytesToBase64(new Uint8Array(data.buffer)),
        },
    };
}

describe('SessionStore', () => {
    test('groups tensors by session and keeps payloads retrievable', () => {
        const store = new SessionStore();
        const summary = store.add('a', tensorMessage('a', 'first', [1, 2, 3]));
        store.add('b', tensorMessage('b', 'other', [4]));

        const sessions = store.summaries();
        expect(sessions.map((s) => s.id).sort()).toEqual(['a', 'b']);
        const payload = store.payload('a', summary.id);
        expect(payload?.name).toBe('first');
        expect(payload?.tensor.shape).toEqual([3]);
        expect(summary.elems).toBe(3);
        expect(summary.bytes).toBe(12);
    });

    test('drops the oldest tensors past the history cap', () => {
        const store = new SessionStore({ maxTensorsPerSession: 2 });
        const first = store.add('a', tensorMessage('a', 'one', [1]));
        store.add('a', tensorMessage('a', 'two', [2]));
        store.add('a', tensorMessage('a', 'three', [3]));

        const session = store.summaries()[0];
        expect(session.tensors.map((t) => t.name)).toEqual(['two', 'three']);
        expect(session.totalReceived).toBe(3);
        expect(store.payload('a', first.id)).toBeNull();
    });

    test('clears one session or all of them', () => {
        const store = new SessionStore();
        store.add('a', tensorMessage('a', 'x', [1]));
        store.add('b', tensorMessage('b', 'y', [1]));
        store.clear('a');
        expect(store.summaries().map((s) => s.id)).toEqual(['b']);
        store.clear();
        expect(store.summaries()).toEqual([]);
    });
});

describe('TensorServer', () => {
    test('stores tensors written to the socket, and honours clear', async () => {
        const store = new SessionStore();
        let changes = 0;
        const server = new TensorServer(store, {
            port: 0, // ephemeral: the test picks whatever the OS gives us
            onChange: () => {
                changes++;
            },
        });
        const info = server.start();
        expect(info.listening).toBe(true);

        const port = (server as unknown as { server: { port: number } }).server.port;
        const socket = await Bun.connect({
            hostname: '127.0.0.1',
            port,
            socket: { data: () => {} },
        });

        // Two tensors and a bad line in one write: the bad line is skipped.
        socket.write(
            `${JSON.stringify(tensorMessage('run-1', 'a', [1, 2]))}\n` +
                'not json\n' +
                `${JSON.stringify(tensorMessage('run-2', 'b', [3]))}\n`
        );
        await waitFor(() => store.summaries().length === 2);
        expect(changes).toBeGreaterThan(0);

        socket.write(`${JSON.stringify({ type: 'clear', session: 'run-1' })}\n`);
        await waitFor(() => store.summaries().length === 1);
        expect(store.summaries()[0].id).toBe('run-2');

        socket.end();
        server.stop();
    });
});

async function waitFor(predicate: () => boolean, timeoutMs = 2000): Promise<void> {
    const deadline = Date.now() + timeoutMs;
    while (!predicate()) {
        if (Date.now() > deadline) {
            throw new Error('condition not met before the timeout');
        }
        await Bun.sleep(10);
    }
}

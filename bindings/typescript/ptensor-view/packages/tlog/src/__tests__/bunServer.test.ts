import { describe, expect, test } from 'bun:test';
import { bytesToBase64 } from 'ptensor-ts';
import { TensorServer } from '../bun/server';
import type { TensorPayload } from '../connectionHandler';
import { DEFAULT_SESSION } from '../constants';

function tensorLine(name: string, values: number[]): string {
    const data = new Float32Array(values);
    const bytes = new Uint8Array(data.buffer);
    return `${JSON.stringify({
        name,
        tensor: {
            dtype: 'float32',
            shape: [values.length],
            stride: [1],
            size_bytes: bytes.byteLength,
            encoding: 'base64',
            blob: bytesToBase64(bytes),
        },
    })}\n`;
}

/** Starts a server on an ephemeral port and connects one producer to it. */
async function connected(received: TensorPayload[]) {
    const server = new TensorServer({ port: 0, onTensor: (p) => received.push(p) });
    expect(server.start().listening).toBe(true);
    const port = (server as unknown as { server: { port: number } }).server.port;
    const socket = await Bun.connect({
        hostname: '127.0.0.1',
        port,
        socket: { data: () => {} },
    });
    return { server, socket };
}

/** Lets the write reach the listener and the handler run. */
const settle = () => Bun.sleep(30);

describe('TensorServer', () => {
    test('pushes tensors under the session the producer announced', async () => {
        const received: TensorPayload[] = [];
        const { server, socket } = await connected(received);

        socket.write(`${JSON.stringify({ sessionId: 'run-a' })}\n`);
        socket.write(tensorLine('first', [1, 2, 3]));
        socket.write(tensorLine('second', [4]));
        await settle();

        expect(received.map((p) => p.name)).toEqual(['first', 'second']);
        expect(received.every((p) => p.sessionId === 'run-a')).toBe(true);
        expect(received[0].tensor.shape).toEqual([3]);
        expect(received[0].receivedAt).toBeGreaterThan(0);

        socket.end();
        server.stop();
    });

    test('falls back to the default session when none is announced', async () => {
        const received: TensorPayload[] = [];
        const { server, socket } = await connected(received);

        socket.write(tensorLine('lonely', [1]));
        await settle();

        expect(received.map((p) => p.sessionId)).toEqual([DEFAULT_SESSION]);

        socket.end();
        server.stop();
    });

    test('skips a malformed line and keeps the connection', async () => {
        const received: TensorPayload[] = [];
        const { server, socket } = await connected(received);

        // Three lines in one write: bad JSON, a message of no known kind, and a
        // good tensor. Only the last one is pushed.
        socket.write(`not json\n${JSON.stringify({ hello: 'world' })}\n${tensorLine('ok', [1])}`);
        await settle();

        expect(received.map((p) => p.name)).toEqual(['ok']);

        socket.end();
        server.stop();
    });

    test('drops the connection on a line over the size cap', async () => {
        const received: TensorPayload[] = [];
        const server = new TensorServer({
            port: 0,
            maxLineBytes: 64,
            onTensor: (p) => received.push(p),
        });
        server.start();
        const port = (server as unknown as { server: { port: number } }).server.port;
        let closed = false;
        const socket = await Bun.connect({
            hostname: '127.0.0.1',
            port,
            socket: { data: () => {}, close: () => { closed = true; } },
        });

        socket.write(`${'x'.repeat(256)}`);
        await settle();

        expect(received).toEqual([]);
        expect(closed).toBe(true);

        server.stop();
    });

    test('reports a bind failure through serverInfo', () => {
        const first = new TensorServer({ port: 0, onTensor: () => {} });
        first.start();
        const port = (first as unknown as { server: { port: number } }).server.port;

        const second = new TensorServer({ port, onTensor: () => {} });
        const info = second.start();
        expect(info.listening).toBe(false);
        expect(info.error).toBeDefined();

        first.stop();
        second.stop();
    });
});

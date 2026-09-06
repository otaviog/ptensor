import { beforeEach, describe, expect, test } from 'bun:test';
import type { TensorJson } from 'ptensor-ts';
import { handleMessage, type TensorPayload } from '../connectionHandler';
import { DEFAULT_SESSION } from '../constants';

const tensor: TensorJson = {
    dtype: 'float32',
    shape: [1],
    stride: [1],
    size_bytes: 4,
    encoding: 'base64',
    blob: 'AAAAAA==',
};

/** Everything the handler reported, newest last. */
const logged: string[] = [];
const logger = {
    info: (message: string) => logged.push(message),
    warn: (message: string) => logged.push(message),
};

beforeEach(() => {
    logged.length = 0;
});

/** A fresh connection's handler, with its tensors collected in `received`. */
function handler(received: TensorPayload[]) {
    return handleMessage({ clientAddress: '127.0.0.1' }, (payload) => received.push(payload), logger);
}

/** Just its message pump, for the tests that do not care about the close log. */
function messagePump(received: TensorPayload[]) {
    return handler(received).onMessage;
}

function closeMessages(): string[] {
    return logged.filter((message) => message.includes('closed connection'));
}

describe('handleMessage', () => {
    test('tags tensors with the session announced by the connection', () => {
        const received: TensorPayload[] = [];
        const handle = messagePump(received);

        handle({ kind: 'session-message', sessionId: 'run-a' });
        handle({ kind: 'tensor-message', name: 'first', tensor });

        expect(received).toHaveLength(1);
        expect(received[0].sessionId).toBe('run-a');
        expect(received[0].name).toBe('first');
        expect(received[0].tensor).toBe(tensor);
    });

    test('uses the default session for a connection that announced none', () => {
        const received: TensorPayload[] = [];
        messagePump(received)({ kind: 'tensor-message', name: 'first', tensor });

        expect(received[0].sessionId).toBe(DEFAULT_SESSION);
    });

    test('keeps the first session id when the producer sends another', () => {
        const received: TensorPayload[] = [];
        const handle = messagePump(received);

        handle({ kind: 'session-message', sessionId: 'run-a' });
        handle({ kind: 'session-message', sessionId: 'run-b' });
        handle({ kind: 'tensor-message', name: 'first', tensor });

        expect(received[0].sessionId).toBe('run-a');
    });

    test('stamps the arrival time', () => {
        const received: TensorPayload[] = [];
        const before = Date.now();
        messagePump(received)({ kind: 'tensor-message', name: 'first', tensor });

        expect(received[0].receivedAt).toBeGreaterThanOrEqual(before);
    });

    test('logs the session id of the connection it closes', () => {
        const received: TensorPayload[] = [];
        const { onMessage, onConnectionClose } = handler(received);

        onMessage({ kind: 'session-message', sessionId: 'run-a' });
        onConnectionClose();

        expect(closeMessages()).toHaveLength(1);
        expect(closeMessages()[0]).toContain('run-a');
        expect(closeMessages()[0]).toContain('127.0.0.1');
    });

    test('logs the close of a connection that announced no session', () => {
        const received: TensorPayload[] = [];
        handler(received).onConnectionClose();

        expect(closeMessages()).toHaveLength(1);
        expect(closeMessages()[0]).toContain('undefined');
    });

    test('reports a producer that tries to change its session', () => {
        const handle = messagePump([]);

        handle({ kind: 'session-message', sessionId: 'run-a' });
        handle({ kind: 'session-message', sessionId: 'run-b' });

        expect(logged.some((m) => m.includes('attempted to change session ID'))).toBe(true);
    });

    test('works with no logger at all', () => {
        const received: TensorPayload[] = [];
        const handler = handleMessage({ clientAddress: '127.0.0.1' }, (p) => received.push(p));

        handler.onMessage({ kind: 'tensor-message', name: 'first', tensor });
        handler.onConnectionClose();

        expect(received).toHaveLength(1);
    });
});

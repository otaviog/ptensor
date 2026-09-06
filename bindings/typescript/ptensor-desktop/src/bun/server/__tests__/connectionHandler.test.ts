import { describe, expect, test } from 'bun:test';
import type { TensorJson } from 'ptensor-ts';
import type { TensorPayload } from '../../../shared/rpc';
import { handleMessage } from '../connectionHandler';
import { DEFAULT_SESSION } from '../constants';

const tensor: TensorJson = {
    dtype: 'float32',
    shape: [1],
    stride: [1],
    size_bytes: 4,
    encoding: 'base64',
    blob: 'AAAAAA==',
};

function handler(received: TensorPayload[]) {
    return handleMessage({ clientAddress: '127.0.0.1' }, (payload) => received.push(payload));
}

describe('handleMessage', () => {
    test('tags tensors with the session announced by the connection', () => {
        const received: TensorPayload[] = [];
        const handle = handler(received);

        handle({ kind: 'session-message', sessionId: 'run-a' });
        handle({ kind: 'tensor-message', name: 'first', tensor });

        expect(received).toHaveLength(1);
        expect(received[0].sessionId).toBe('run-a');
        expect(received[0].name).toBe('first');
        expect(received[0].tensor).toBe(tensor);
    });

    test('uses the default session for a connection that announced none', () => {
        const received: TensorPayload[] = [];
        handler(received)({ kind: 'tensor-message', name: 'first', tensor });

        expect(received[0].sessionId).toBe(DEFAULT_SESSION);
    });

    test('keeps the first session id when the producer sends another', () => {
        const received: TensorPayload[] = [];
        const handle = handler(received);

        handle({ kind: 'session-message', sessionId: 'run-a' });
        handle({ kind: 'session-message', sessionId: 'run-b' });
        handle({ kind: 'tensor-message', name: 'first', tensor });

        expect(received[0].sessionId).toBe('run-a');
    });

    test('stamps the arrival time', () => {
        const received: TensorPayload[] = [];
        const before = Date.now();
        handler(received)({ kind: 'tensor-message', name: 'first', tensor });

        expect(received[0].receivedAt).toBeGreaterThanOrEqual(before);
    });
});

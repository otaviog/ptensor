import { afterAll, beforeAll, beforeEach, describe, expect, test } from 'bun:test';
import { configureSync, getConsoleSink, type LogRecord } from '@logtape/logtape';
import type { TensorJson } from 'ptensor-ts';
import { configureLogging } from '../../../shared/logging';
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

/** A fresh connection's handler, with its tensors collected in `received`. */
function handler(received: TensorPayload[]) {
    return handleMessage({ clientAddress: '127.0.0.1' }, (payload) => received.push(payload));
}

/** Just its message pump, for the tests that do not care about the close log. */
function messagePump(received: TensorPayload[]) {
    return handler(received).onMessage;
}

const records: LogRecord[] = [];

beforeAll(() => {
    // Closing a connection only logs, so capture the records rather than the
    // console. `configureLogging` runs first to claim the module's one-shot
    // flag, otherwise the handler's own `getAppLogger` would reset the sinks
    // back to the console on the first log line.
    configureLogging();
    configureSync({
        sinks: { capture: (record: LogRecord) => records.push(record) },
        loggers: [
            { category: 'ptensor-desktop', sinks: ['capture'], lowestLevel: 'debug' },
            { category: ['logtape', 'meta'], sinks: ['capture'], lowestLevel: 'warning' },
        ],
        reset: true,
    });
});

afterAll(() => {
    configureSync({
        sinks: { console: getConsoleSink() },
        loggers: [
            { category: 'ptensor-desktop', sinks: ['console'], lowestLevel: 'info' },
            { category: ['logtape', 'meta'], sinks: ['console'], lowestLevel: 'warning' },
        ],
        reset: true,
    });
});

beforeEach(() => {
    records.length = 0;
});

function closeRecords(): LogRecord[] {
    return records.filter((record) => record.rawMessage.includes('closed connection'));
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

        const [record] = closeRecords();
        expect(closeRecords()).toHaveLength(1);
        expect(record.properties.sessionId).toBe('run-a');
        expect(record.properties.connection).toBe('127.0.0.1');
    });

    test('logs the close of a connection that announced no session', () => {
        const received: TensorPayload[] = [];
        handler(received).onConnectionClose();

        const [record] = closeRecords();
        expect(closeRecords()).toHaveLength(1);
        expect(record.properties.sessionId).toBeUndefined();
        expect(record.properties.connection).toBe('127.0.0.1');
    });
});

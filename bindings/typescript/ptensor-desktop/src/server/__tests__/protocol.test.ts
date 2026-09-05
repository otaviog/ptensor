import { describe, expect, test } from 'bun:test';
import { bytesToBase64, type TensorJson } from 'ptensor-ts';
import { parseIncoming, type SessionMessage, type TensorMessage } from '../protocol';
import { PdError } from '../../shared/pdError';

function tensorJson(values: number[]): TensorJson {
    const data = new Float32Array(values);
    return {
        dtype: 'float32',
        shape: [values.length],
        stride: [1],
        blob: bytesToBase64(new Uint8Array(data.buffer)),
    };
}

describe('parseIncoming', () => {
    test('rejects anything that is not a JSON object', () => {
        for (const value of [null, undefined, 42, 'hello', true]) {
            expect(() => parseIncoming(value)).toThrow(PdError);
        }
        expect(() => parseIncoming(null)).toThrow('message is not a JSON object');
    });

    test('accepts a session message', () => {
        const message = parseIncoming({ sessionId: 'abc' }) as SessionMessage;
        expect(message.sessionId).toBe('abc');
    });

    test('reads a session message before a tensor message', () => {
        // Both fields present: sessionId decides, so the tensor is not validated.
        const message = parseIncoming({ sessionId: 'abc', name: 'x', tensor: {} });
        expect(message).toEqual({ sessionId: 'abc', name: 'x', tensor: {} } as never);
    });

    test('rejects a session message whose id is not a string', () => {
        expect(() => parseIncoming({ sessionId: 7 })).toThrow(
            'message is neither a session message nor a tensor message',
        );
    });

    test('accepts a tensor message and keeps only name and tensor', () => {
        const tensor = tensorJson([1, 2, 3]);
        const message = parseIncoming({ name: 'first', tensor, extra: 'ignored' }) as TensorMessage;
        expect(message).toEqual({ name: 'first', tensor });
    });

    test('rejects a tensor message with a missing or non-string name', () => {
        const tensor = tensorJson([1]);
        expect(() => parseIncoming({ tensor })).toThrow(PdError);
        expect(() => parseIncoming({ name: 7, tensor })).toThrow(
            'message is neither a session message nor a tensor message',
        );
    });

    test('rejects a tensor message whose tensor is not an object', () => {
        expect(() => parseIncoming({ name: 'first', tensor: 'not-a-tensor' })).toThrow(
            'message is neither a session message nor a tensor message',
        );
    });

    test('wraps a tensor validation failure in a PdError', () => {
        // `typeof null === 'object'`, so this reaches validateTensorJson.
        expect(() => parseIncoming({ name: 'first', tensor: null })).toThrow(PdError);

        const { dtype, ...noDtype } = tensorJson([1]);
        expect(() => parseIncoming({ name: 'first', tensor: noDtype })).toThrow(
            /^P10Error: .*'dtype'/,
        );
    });

    test('reports each missing tensor field', () => {
        const fields = ['dtype', 'shape', 'stride', 'blob'] as const;
        for (const field of fields) {
            const tensor: Record<string, unknown> = { ...tensorJson([1]) };
            delete tensor[field];
            expect(() => parseIncoming({ name: 'first', tensor })).toThrow(
                new RegExp(`^P10Error: .*'${field}'`),
            );
        }
    });
});

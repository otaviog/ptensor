// The decode worker, for real: the bundle the build step produced, started from
// a blob URL the way the window starts it, fetching from a real feed server.
//
// bun runs blob-URL module workers, so this covers the bundle parsing, the
// message protocol and the buffer transfer -- everything the window does except
// being a window.

import { afterEach, describe, expect, test } from 'bun:test';
import { TensorStore } from 'ptensor-tlog';
import type { FeedInfo, TensorPayload } from 'ptensor-tlog';
import type { TensorJson } from 'ptensor-ts';
import { FeedServer } from '../../bun/feedServer';
import { createDecoder, createInlineDecoder, type TensorDecoder } from '../decoder';
import { fromTransfer, toTransfer } from '../decodeTransfer';

const TOKEN = 'd'.repeat(64);
const INFO: FeedInfo = { host: '127.0.0.1', port: 4449, listening: true };

function jsonOf(dtype: string, shape: number[], bytes: Uint8Array): TensorJson {
    return {
        dtype,
        shape,
        stride: [1],
        size_bytes: bytes.byteLength,
        encoding: 'base64+zstd',
        blob: Buffer.from(Bun.zstdCompressSync(bytes)).toString('base64'),
    };
}

function float32Json(values: number[]): TensorJson {
    const data = new Float32Array(values);
    return jsonOf('float32', [values.length], new Uint8Array(data.buffer));
}

function payload(name: string, json: TensorJson): TensorPayload {
    return { sessionId: 'run-a', name, tensor: json, receivedAt: Date.now() };
}

let running: FeedServer | null = null;
let decoder: TensorDecoder | null = null;

afterEach(() => {
    decoder?.dispose();
    decoder = null;
    running?.stop();
    running = null;
});

function start(store: TensorStore) {
    const server = new FeedServer({ store, token: TOKEN, feedInfo: () => INFO });
    running = server;
    const address = server.start();
    return (id: string) => `${address.origin}/tensor/${id}?token=${TOKEN}`;
}

describe('toTransfer / fromTransfer', () => {
    test('round trips a tensor through the transferable form', () => {
        const transfer = toTransfer(float32Json([1, -2, 0.5, 4]));
        const tensor = fromTransfer(transfer);

        expect(tensor.dtype).toBe('float32');
        expect(tensor.shape).toEqual([4]);
        expect([...tensor.data]).toEqual([1, -2, 0.5, 4]);
    });

    test('keeps float16 as the tensor dtype while carrying it widened', () => {
        // 1.0, -2.0, 0.5 as IEEE half.
        const halves = new Uint16Array([0x3c00, 0xc000, 0x3800]);
        const transfer = toTransfer(jsonOf('float16', [3], new Uint8Array(halves.buffer)));

        // The buffer holds float32 -- there is no float16 typed array -- so the
        // view dtype and the tensor's own dtype have to differ.
        expect(transfer.dtype).toBe('float16');
        expect(transfer.viewDtype).toBe('float32');
        expect(transfer.buffer.byteLength).toBe(12);

        const tensor = fromTransfer(transfer);
        expect(tensor.dtype).toBe('float16');
        expect(tensor.data).toBeInstanceOf(Float32Array);
        expect([...tensor.data]).toEqual([1, -2, 0.5]);
    });
});

describe('createDecoder', () => {
    test('decodes in the worker and transfers the buffer back', async () => {
        const store = new TensorStore();
        const url = start(store);
        const { meta } = store.add(payload('frame', float32Json([1, -2, 0.5, 4])));
        decoder = createDecoder();

        const tensor = await decoder.decode(url(meta.id));

        // Asserted, because the fallback would pass every other expectation
        // here and the only sign would be a line in the log.
        expect(decoder.usesWorker).toBe(true);
        expect(tensor.dtype).toBe('float32');
        expect([...tensor.data]).toEqual([1, -2, 0.5, 4]);
        // Transferred, not copied: the buffer belongs to this thread now.
        expect(tensor.data.buffer.byteLength).toBe(16);
    });

    test('keeps one worker across several decodes, in any order', async () => {
        const store = new TensorStore();
        const url = start(store);
        const first = store.add(payload('first', float32Json([1, 2]))).meta;
        const second = store.add(payload('second', float32Json([3, 4, 5]))).meta;
        decoder = createDecoder();

        // Both in flight at once: the ids are what pairs answer to caller.
        const [a, b] = await Promise.all([
            decoder.decode(url(first.id)),
            decoder.decode(url(second.id)),
        ]);

        expect(decoder.usesWorker).toBe(true);
        expect([...a.data]).toEqual([1, 2]);
        expect([...b.data]).toEqual([3, 4, 5]);
    });

    test('reports a failed fetch through the worker', async () => {
        const url = start(new TensorStore());
        const own = createDecoder();
        decoder = own;

        await expect(own.decode(url('gone'))).rejects.toThrow(/404/);
        expect(own.usesWorker).toBe(true);
    });

    test('rejects what is still in flight when disposed', async () => {
        const store = new TensorStore();
        const url = start(store);
        const { meta } = store.add(payload('frame', float32Json([1])));
        const own = createDecoder();

        const pending = own.decode(url(meta.id));
        own.dispose();

        await expect(pending).rejects.toThrow(/disposed/);
    });

    test('the inline decoder is the same contract without a worker', async () => {
        const store = new TensorStore();
        const url = start(store);
        const { meta } = store.add(payload('frame', float32Json([7, 8])));
        const own = createInlineDecoder();
        decoder = own;

        expect(own.usesWorker).toBe(false);
        expect([...(await own.decode(url(meta.id))).data]).toEqual([7, 8]);
    });
});

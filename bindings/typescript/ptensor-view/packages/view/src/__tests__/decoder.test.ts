// The decode worker, for real: the bundle the build step produced, started from
// a blob URL the way a host starts it, fetching over a real socket.
//
// bun runs blob-URL module workers, so this covers the bundle parsing, the
// message protocol and the buffer transfer -- everything a host does except
// being a host. The server here is a stub on purpose: the decoder's contract is
// a URL that answers with a `TensorJson`, and nothing more.

import { afterEach, describe, expect, test } from 'bun:test';
import type { TensorJson } from 'ptensor-ts';
import { createDecoder, createInlineDecoder, type TensorDecoder } from '../decode/decoder';
import { fromTransfer, toTransfer } from '../decode/decodeTransfer';

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

let server: Bun.Server<undefined> | null = null;
let decoder: TensorDecoder | null = null;

afterEach(() => {
    decoder?.dispose();
    decoder = null;
    server?.stop(true);
    server = null;
});

/** Serves the given tensors by name; anything else is a 404. */
function serve(tensors: Record<string, TensorJson>) {
    const running = Bun.serve({
        hostname: '127.0.0.1',
        port: 0,
        fetch(request) {
            const name = new URL(request.url).pathname.slice(1);
            const tensor = tensors[name];
            return tensor === undefined
                ? new Response('no such tensor', { status: 404 })
                : Response.json(tensor);
        },
    });
    server = running;
    return (name: string) => `http://127.0.0.1:${running.port}/${name}`;
}

describe('toTransfer / fromTransfer', () => {
    test('round trips a tensor through the transferable form', () => {
        const tensor = fromTransfer(toTransfer(float32Json([1, -2, 0.5, 4])));

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
        const url = serve({ frame: float32Json([1, -2, 0.5, 4]) });
        decoder = createDecoder();

        const tensor = await decoder.decode(url('frame'));

        // Asserted, because the fallback would pass every other expectation
        // here and the only sign would be a line in the log.
        expect(decoder.usesWorker).toBe(true);
        expect(tensor.dtype).toBe('float32');
        expect([...tensor.data]).toEqual([1, -2, 0.5, 4]);
        // Transferred, not copied: the buffer belongs to this thread now.
        expect(tensor.data.buffer.byteLength).toBe(16);
    });

    test('keeps one worker across several decodes, in any order', async () => {
        const url = serve({ first: float32Json([1, 2]), second: float32Json([3, 4, 5]) });
        decoder = createDecoder();

        // Both in flight at once: the request id is what pairs an answer to its
        // caller.
        const [a, b] = await Promise.all([
            decoder.decode(url('first')),
            decoder.decode(url('second')),
        ]);

        expect(decoder.usesWorker).toBe(true);
        expect([...a.data]).toEqual([1, 2]);
        expect([...b.data]).toEqual([3, 4, 5]);
    });

    test('reports a failed fetch through the worker', async () => {
        const url = serve({});
        const own = createDecoder();
        decoder = own;

        await expect(own.decode(url('gone'))).rejects.toThrow(/404/);
        expect(own.usesWorker).toBe(true);
    });

    test('tells the host when it has fallen back to this thread', async () => {
        const url = serve({ frame: float32Json([1, 2]) });
        const warnings: string[] = [];
        const own = createDecoder({ warn: (message) => warnings.push(message) });
        decoder = own;

        // No `Worker` to be had: the decode still has to produce the tensor.
        const realWorker = globalThis.Worker;
        try {
            (globalThis as { Worker?: unknown }).Worker = undefined;
            expect([...(await own.decode(url('frame'))).data]).toEqual([1, 2]);
        } finally {
            globalThis.Worker = realWorker;
        }

        expect(own.usesWorker).toBe(false);
        expect(warnings.join(' ')).toContain('calling thread');
    });

    test('rejects what is still in flight when disposed', async () => {
        const url = serve({ frame: float32Json([1]) });
        const own = createDecoder();

        const pending = own.decode(url('frame'));
        own.dispose();

        await expect(pending).rejects.toThrow(/disposed/);
    });

    test('the inline decoder is the same contract without a worker', async () => {
        const url = serve({ frame: float32Json([7, 8]) });
        const own = createInlineDecoder();
        decoder = own;

        expect(own.usesWorker).toBe(false);
        expect([...(await own.decode(url('frame'))).data]).toEqual([7, 8]);
    });
});

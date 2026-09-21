// The decode worker. Bundled to a string by scripts/buildDecodeWorker.ts and
// started from a blob URL (see ./decoder.ts), because the bundler leaves
// `new Worker(new URL(...))` alone and emits no worker chunk.
//
// It does the fetch itself, not just the decode. That keeps the tensor's base64
// -- 262 MiB for a batched float32 image -- out of the window's heap entirely:
// it is read, decoded and dropped here, and only the decoded buffer crosses,
// transferred rather than copied.

import { fetchTensorJson, toTransfer, type DecodeRequest } from './decodeTransfer';

declare const self: {
    onmessage: ((event: MessageEvent<DecodeRequest>) => void) | null;
    postMessage(message: unknown, transfer?: Transferable[]): void;
};

self.onmessage = async (event: MessageEvent<DecodeRequest>) => {
    const { id, url } = event.data;
    try {
        const transfer = toTransfer(await fetchTensorJson(url));
        self.postMessage({ id, ok: true, ...transfer }, [transfer.buffer]);
    } catch (error: unknown) {
        self.postMessage({
            id,
            ok: false,
            error: error instanceof Error ? error.message : String(error),
        });
    }
};

// Decoding, off the calling thread.
//
// base64 plus zstd on a batched float32 image is about 1.3 s of straight-line
// JS. On a UI thread that is 1.3 s of frozen panel per tensor the user clicks,
// so it runs in a worker and the decoded buffer is transferred back.
//
// The worker is started from a blob URL built over a source string the build
// step generated (./.generated/decodeWorkerSource). If it cannot start -- no
// `Worker`, blob URLs refused, a bundle the engine will not parse -- decoding
// falls back to the calling thread, which is slow but correct, rather than
// failing.

import type { Tensor } from 'ptensor-ts';
import { DECODE_WORKER_SOURCE } from './.generated/decodeWorkerSource';
import {
    fetchTensorJson,
    fromTransfer,
    toTransfer,
    type DecodeRequest,
    type DecodeResponse,
} from './decodeTransfer';
/**
 * Where the decoder says what happened. The host owns logging, so this is the
 * whole of it -- a `console` satisfies it, and so does a real logger.
 */
export interface DecoderLogger {
    debug?(message: string): void;
    warn?(message: string): void;
}

export interface TensorDecoder {
    /** Fetches and decodes the tensor at `url`. */
    decode(url: string): Promise<Tensor>;
    /**
     * Whether decodes are still going off-thread. False once the worker has
     * been given up on -- worth knowing, because the only symptom otherwise is
     * a panel that stutters.
     */
    readonly usesWorker: boolean;
    /** Stops the worker. Pending decodes reject. */
    dispose(): void;
}

interface Waiting {
    resolve: (tensor: Tensor) => void;
    reject: (error: Error) => void;
    url: string;
}

/** Decodes on this thread. The fallback, and what the worker does internally. */
async function decodeHere(url: string): Promise<Tensor> {
    return fromTransfer(toTransfer(await fetchTensorJson(url)));
}

/**
 * A decoder that never starts a worker: the fallback path on its own. Used by
 * tests, which have no engine to run a worker in.
 */
export function createInlineDecoder(): TensorDecoder {
    return { decode: decodeHere, usesWorker: false, dispose: () => {} };
}

export function createDecoder(log: DecoderLogger = {}): TensorDecoder {
    const waiting = new Map<number, Waiting>();
    let worker: Worker | null = null;
    let broken = false;
    let nextId = 1;
    let disposed = false;
    // Revoked on dispose, not right after construction: the worker fetches its
    // script asynchronously, so revoking immediately races it -- bun refuses
    // outright with "Blob URL is missing", and a browser is not obliged to do
    // better.
    let sourceUrl: string | null = null;

    /** Hands every outstanding decode to this thread and stops using the worker. */
    const giveUpOnWorker = (why: string): void => {
        if (!broken) {
            broken = true;
            log.warn?.(`Decoding on the calling thread: ${why}.`);
        }
        stopWorker();
        for (const [id, pending] of waiting) {
            waiting.delete(id);
            decodeHere(pending.url).then(pending.resolve, pending.reject);
        }
    };

    const stopWorker = (): void => {
        worker?.terminate();
        worker = null;
        if (sourceUrl !== null) {
            URL.revokeObjectURL(sourceUrl);
            sourceUrl = null;
        }
    };

    const onMessage = (event: MessageEvent<DecodeResponse>): void => {
        const response = event.data;
        const pending = waiting.get(response.id);
        if (pending === undefined) {
            return;
        }
        waiting.delete(response.id);
        if (response.ok) {
            try {
                pending.resolve(fromTransfer(response));
            } catch (error: unknown) {
                pending.reject(error instanceof Error ? error : new Error(String(error)));
            }
        } else {
            pending.reject(new Error(response.error));
        }
    };

    const start = (): Worker | null => {
        if (worker !== null || broken || disposed) {
            return worker;
        }
        try {
            const blob = new Blob([DECODE_WORKER_SOURCE], { type: 'text/javascript' });
            sourceUrl = URL.createObjectURL(blob);
            const started = new Worker(sourceUrl, { type: 'module' });
            started.addEventListener('message', onMessage as EventListener);
            // A bundle the engine refuses shows up here, not as a throw above.
            started.addEventListener('error', (event: ErrorEvent) => {
                giveUpOnWorker(event.message || 'the decode worker failed to start');
            });
            worker = started;
            log.debug?.('Decode worker started.');
        } catch (error: unknown) {
            giveUpOnWorker(error instanceof Error ? error.message : String(error));
        }
        return worker;
    };

    return {
        decode(url) {
            const active = start();
            if (active === null) {
                return decodeHere(url);
            }
            return new Promise<Tensor>((resolve, reject) => {
                const id = nextId++;
                waiting.set(id, { resolve, reject, url });
                const request: DecodeRequest = { id, url };
                active.postMessage(request);
            });
        },

        get usesWorker() {
            return !broken;
        },

        dispose() {
            disposed = true;
            stopWorker();
            for (const [id, pending] of waiting) {
                waiting.delete(id);
                pending.reject(new Error('the decoder was disposed'));
            }
        },
    };
}

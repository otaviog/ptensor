// Decoded tensors the window is holding on to.
//
// Decoding a tensor means allocating its uncompressed bytes -- a batched
// float32 image is a couple of hundred megabytes -- so what the window keeps is
// capped in bytes and the least recently looked at goes first. Without that,
// clicking through a session's history is a straight line to an out-of-memory
// window.

import type { Tensor } from '@ptensor/tensor-view';

/** 1 GiB of decoded tensors: a few big ones, or a lot of ordinary ones. */
export const DEFAULT_DECODED_BUDGET_BYTES = 1024 * 1024 * 1024;

export class TensorCache {
    /** Least recently used first: `get` and `put` move an entry to the end. */
    private readonly entries = new Map<string, Tensor>();
    private used = 0;

    constructor(private readonly budgetBytes: number = DEFAULT_DECODED_BUDGET_BYTES) {}

    /** Decoded bytes currently held. */
    get usedBytes(): number {
        return this.used;
    }

    get size(): number {
        return this.entries.size;
    }

    get(id: string): Tensor | undefined {
        const tensor = this.entries.get(id);
        if (tensor === undefined) {
            return undefined;
        }
        this.entries.delete(id);
        this.entries.set(id, tensor);
        return tensor;
    }

    /**
     * Files a decoded tensor. The newest is kept even if it is alone over the
     * budget: it is the one on screen.
     */
    put(id: string, tensor: Tensor): void {
        this.delete(id);
        this.entries.set(id, tensor);
        this.used += tensor.data.byteLength;

        for (const [candidate, held] of this.entries) {
            if (this.used <= this.budgetBytes) {
                break;
            }
            if (candidate === id) {
                continue;
            }
            this.used -= held.data.byteLength;
            this.entries.delete(candidate);
        }
    }

    delete(id: string): void {
        const held = this.entries.get(id);
        if (held !== undefined) {
            this.used -= held.data.byteLength;
            this.entries.delete(id);
        }
    }

    clear(): void {
        this.entries.clear();
        this.used = 0;
    }
}

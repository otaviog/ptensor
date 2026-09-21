// What a viewer host keeps between the feed and whatever shows the tensors.
//
// A tensor is held as the `TensorJson` text it arrived as -- still base64, still
// zstd-compressed -- so answering a request is handing back a string, with no
// decode and no per-request work. The decoding belongs to the client that
// actually draws the tensor.
//
// The budget is in bytes, not in tensors: one batched float32 image is a couple
// of hundred megabytes and a hundred of them cannot be held whatever the count
// says.

import type { TensorJson } from 'ptensor-ts';
import type { TensorPayload } from './connectionHandler';

/** Everything about a stored tensor except the blob. What `/tensors` lists. */
export interface TensorMeta {
  /** URL-safe, unique for the life of the store. Derived from `name`. */
  id: string;
  sessionId: string;
  /** Label the producer sent. Not unique: producers log in loops. */
  name: string;
  dtype: string;
  shape: number[];
  stride: number[];
  /** Uncompressed element bytes, so a client can size its buffer. */
  size_bytes: number;
  encoding: string;
  /** Bytes of JSON text the store holds for this tensor. */
  storedBytes: number;
  receivedAt: number;
}

/** What an `add` did: the tensor filed, and whatever had to go to fit it. */
export interface AddResult {
  meta: TensorMeta;
  /**
   * Tensors dropped to stay under the budget. A host that mirrors the store --
   * a window listing what is held -- has to hear about these, so they are part
   * of the result rather than something to notice later on a 404.
   */
  evicted: TensorMeta[];
}

/** Where the store reports evictions; a host with no logger passes nothing. */
export interface StoreLogger {
  info?(message: string): void;
  warn?(message: string): void;
}

export interface TensorStoreOptions {
  /**
   * JSON text the store holds before it drops the oldest tensors, in bytes.
   * A single tensor over the budget is still kept -- dropping the thing the
   * user just asked to see would be worse than briefly going over.
   */
  budgetBytes?: number;
  logger?: StoreLogger;
}

/** 2 GiB of held JSON: a handful of batched images, or a long run of frames. */
export const DEFAULT_BUDGET_BYTES = 2 * 1024 * 1024 * 1024;

interface Entry {
  meta: TensorMeta;
  json: string;
}

export class TensorStore {
  private readonly entries = new Map<string, Entry>();
  private readonly slugCounts = new Map<string, number>();
  private readonly budgetBytes: number;
  private readonly log: StoreLogger;
  private used = 0;

  constructor(options: TensorStoreOptions = {}) {
    this.budgetBytes = options.budgetBytes ?? DEFAULT_BUDGET_BYTES;
    this.log = options.logger ?? {};
  }

  /** Bytes of JSON text currently held. */
  get usedBytes(): number {
    return this.used;
  }

  /** Files one received tensor and evicts whatever no longer fits. */
  add(payload: TensorPayload): AddResult {
    const json = JSON.stringify(payload.tensor);
    const meta: TensorMeta = {
      id: this.mintId(payload.name),
      sessionId: payload.sessionId,
      name: payload.name,
      dtype: payload.tensor.dtype,
      shape: payload.tensor.shape.map(Number),
      stride: payload.tensor.stride.map(Number),
      size_bytes: payload.tensor.size_bytes,
      encoding: payload.tensor.encoding,
      storedBytes: json.length,
      receivedAt: payload.receivedAt,
    };

    this.entries.set(meta.id, { meta, json });
    this.used += meta.storedBytes;
    return { meta, evicted: this.evictToBudget(meta.id) };
  }

  /** Newest first, the order the sidebar lists them in. */
  list(): TensorMeta[] {
    return [...this.entries.values()].map((entry) => entry.meta).reverse();
  }

  meta(id: string): TensorMeta | undefined {
    return this.entries.get(id)?.meta;
  }

  /** The stored `TensorJson` text, ready to serve verbatim. */
  json(id: string): string | undefined {
    return this.entries.get(id)?.json;
  }

  /** Drops one session's tensors, or every tensor when given nothing. */
  clear(sessionId?: string): void {
    for (const [id, entry] of this.entries) {
      if (sessionId === undefined || entry.meta.sessionId === sessionId) {
        this.used -= entry.meta.storedBytes;
        this.entries.delete(id);
      }
    }
  }

  /**
   * Drops the oldest tensors until the budget is met. `keep` is the tensor that
   * was just added: it survives even alone over the budget, so a viewer asked
   * to show a huge tensor still gets it.
   */
  private evictToBudget(keep: string): TensorMeta[] {
    const evicted: TensorMeta[] = [];
    for (const [id, entry] of this.entries) {
      if (this.used <= this.budgetBytes) {
        break;
      }
      if (id === keep) {
        continue;
      }
      this.used -= entry.meta.storedBytes;
      this.entries.delete(id);
      evicted.push(entry.meta);
      this.log.info?.(
        `Evicted tensor ${id} (${entry.meta.storedBytes} bytes) to stay under the budget.`
      );
    }
    if (this.used > this.budgetBytes) {
      this.log.warn?.(
        `Tensor ${keep} alone is ${this.used} bytes, over the ${this.budgetBytes} byte budget.`
      );
    }
    return evicted;
  }

  /**
   * A URL-safe id from the producer's label. Names repeat -- logging `frame` in
   * a loop is the normal case -- so a taken slug gets a counter. Counters are
   * never reused, not even after an eviction, so a stale URL cannot come back
   * pointing at a different tensor.
   */
  private mintId(name: string): string {
    const slug = toSlug(name);
    const used = this.slugCounts.get(slug) ?? 0;
    this.slugCounts.set(slug, used + 1);
    return used === 0 ? slug : `${slug}-${used + 1}`;
  }
}

function toSlug(name: string): string {
  const slug = name
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return slug.length > 0 ? slug : 'tensor';
}

/** Re-exported so a host has one import for the store and what it holds. */
export type { TensorJson };

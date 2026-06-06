// ptensor-ts `Tensor` is the materialized, JS-owned form (plain data, no native
// handle). Aliased to avoid clashing with the legacy handle interface below.
import type { NumericArray, Tensor as TensorData } from 'ptensor-ts';
import type { DTypeString } from './dtype';
import type { TypedArrayType } from './typedArray';

export type { DTypeString, NumericArray, TensorData, TypedArrayType };

/** Opaque handle to a native tensor. */
export type TensorHandle = bigint;

/**
 * Handle-first view of a native tensor (bun-FFI backend). The bytes stay in C;
 * the handle is the working object. `data()` is a zero-copy view valid only
 * until `destroy()`; `toTensor()` is an explicit copy into JS-owned memory.
 */
export interface PTensor {
  /** Dimension sizes, cached at construction. */
  readonly shape: number[];
  /** Per-element strides (element counts), cached at construction. */
  readonly stride: number[];
  /** Element data type, cached at construction. */
  readonly dtype: DTypeString;
  /** Total number of elements. */
  size(): bigint;
  /** Total data size in bytes. */
  sizeBytes(): bigint;
  /** Number of dimensions. */
  ndim(): number;
  /** True when the tensor has no elements. */
  isEmpty(): boolean;
  /**
   * Zero-copy typed-array view over the native buffer. Valid only while this
   * PTensor is alive — do not retain it past `destroy()`.
   */
  data(): NumericArray;
  /** Copies the native data out into a JS-owned, materialized `TensorData`. */
  toTensor(): TensorData;
  /** Releases the native tensor handle. Must be called when done. */
  destroy(): void;
}

/**
 * Legacy handle interface implemented by the WASM backend (and historically the
 * bun-FFI backend). New bun-FFI code uses `PTensor`; WASM consolidation onto
 * `PTensor` is deferred.
 */
export interface Tensor {
  /** Total number of elements. */
  getSize(): bigint;
  /** Total data size in bytes. */
  getSizeBytes(): bigint;
  /** Dimension sizes. */
  getShape(): bigint[];
  /** Per-element strides (in element counts, not bytes). */
  getStride(): bigint[];
  /** Number of dimensions. */
  getNdim(): number;
  /** Element data type. */
  getDtype(): DTypeString;
  /** True when the tensor has no elements. */
  isEmpty(): boolean;
  /** Releases the native tensor handle. Must be called when done. */
  delete(): void;
}

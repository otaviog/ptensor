import type { DTypeString } from './dtype';
import type { TypedArrayType } from './typedArray';

export type { DTypeString, TypedArrayType };

/** Opaque handle to a native tensor. */
export type TensorHandle = bigint;

/** Common interface implemented by both the Bun-FFI and WASM backends. */
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

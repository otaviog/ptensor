/**
 * Core data type of the tensor-view module. A `TensorView` is a *decoded*,
 * plain-data tensor: the numeric payload already lives in a JS typed array.
 * It is deliberately free of any native / FFI dependency so it can be shared
 * by the VS Code webview, the Electron pilot, and the Vite dev playground.
 *
 * The dtype vocabulary, the typed-array union, and the per-element byte sizes
 * are owned by ptensor-ts (the dependency-free core); re-exported here so the
 * view's existing import sites keep working.
 */

import { type DTypeString, dtypeSizeBytes, type NumericArray } from 'ptensor-ts';

export type { DTypeString, NumericArray };

export interface TensorView {
    /** Decoded payload. float16 is decoded to Float32Array; int64 to BigInt64Array. */
    array: NumericArray;
    /** Per-element strides (element counts, not bytes). */
    stride: bigint[];
    /** Dimension sizes. */
    shape: bigint[];
    /** Element data type. */
    dtype: DTypeString;
    /** Optional label for the panel header (e.g. the debugger expression). */
    name?: string;
}

/** Size in bytes of one element of each dtype (float16 measured on the wire). */
export const DTYPE_SIZES: Record<DTypeString, number> = dtypeSizeBytes;

/** True for floating dtypes, which need min/max stretching to display as images. */
export function isFloatDtype(dtype: DTypeString): boolean {
    return dtype === 'float32' || dtype === 'float64' || dtype === 'float16';
}

/** Reads element `i` of a NumericArray as a plain number (handles BigInt64Array). */
export function elementAt(array: NumericArray, i: number): number {
    return array instanceof BigInt64Array ? Number(array[i]) : array[i];
}

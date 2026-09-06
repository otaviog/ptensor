import { type DTypeString } from './dtype';
import { type NumericArray } from './numericArray';

/**
 * A materialized, JS-owned tensor: plain data with no native handle. This is
 * what `parse`/`tensorFromJson` produce and what the viewer consumes. The FFI
 * binding's `PTensor.toTensor()` also yields this (an explicit copy out of
 * native memory). Strides are in element counts.
 */
export interface Tensor {
  dtype: DTypeString;
  shape: number[];
  stride: number[];
  data: NumericArray;
}

/** Row-major (C-contiguous) strides for a shape, in element counts. */
export function contiguousStride(shape: number[]): number[] {
  const stride = new Array<number>(shape.length);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    stride[i] = acc;
    acc *= shape[i];
  }
  return stride;
}

/** Element count for a shape. */
export function numElements(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

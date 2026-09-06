import { type DTypeString } from './dtype';
import { type NumericArray, viewNumericArray } from './numericArray';

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

/** True for floating dtypes, which readers stretch to a display range. */
export function isFloatDtype(dtype: DTypeString): boolean {
  return dtype === 'float32' || dtype === 'float64' || dtype === 'float16';
}

/**
 * Reads element `i` of a `NumericArray` as a plain number. int64 arrives as a
 * `BigInt64Array`, whose elements are bigints; everything past 2^53 loses
 * precision here, which is the price of a number.
 */
export function elementAt(array: NumericArray, i: number): number {
  return array instanceof BigInt64Array ? Number(array[i]) : array[i];
}

/**
 * Wraps decoded bytes as the `NumericArray` for `dtype`. float16 is widened to
 * a `Float32Array`, since there is no native float16 typed array and readers
 * want numbers rather than raw bits; the owning `Tensor.dtype` still says
 * `float16`. Every other dtype is a plain view over `buffer`, no copy.
 */
export function bytesToTyped(buffer: ArrayBuffer, dtype: DTypeString): NumericArray {
  if (dtype === 'float16') {
    return float16ToFloat32(new Uint16Array(buffer));
  }
  return viewNumericArray(dtype, buffer);
}

/** IEEE 754 half -> single, including subnormals, infinities and NaN. */
function float16ToFloat32(input: Uint16Array): Float32Array {
  const out = new Float32Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const h = input[i];
    const sign = (h & 0x8000) >> 15;
    const exp = (h & 0x7c00) >> 10;
    const frac = h & 0x03ff;
    if (exp === 0) {
      out[i] = (sign ? -1 : 1) * 2 ** -14 * (frac / 1024);
    } else if (exp === 31) {
      out[i] = frac ? NaN : sign ? -Infinity : Infinity;
    } else {
      out[i] = (sign ? -1 : 1) * 2 ** (exp - 15) * (1 + frac / 1024);
    }
  }
  return out;
}

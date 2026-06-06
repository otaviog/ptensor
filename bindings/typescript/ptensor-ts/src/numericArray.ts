import { type DTypeString } from './dtype';

/**
 * The JS-owned backing store for a materialized tensor. One typed-array variant
 * per dtype. float16 has no native typed array, so its bits are carried in a
 * `Uint16Array` and the owning `Tensor.dtype` keeps the real type tag.
 */
export type NumericArray =
  | Float32Array
  | Float64Array
  | Uint8Array
  | Uint16Array
  | Uint32Array
  | Int8Array
  | Int16Array
  | Int32Array
  | BigInt64Array;

/** Allocates a zero-filled NumericArray of `size` elements for `dtype`. */
export function createNumericArray(dtype: DTypeString, size: number): NumericArray {
  switch (dtype) {
    case 'float32':
      return new Float32Array(size);
    case 'float64':
      return new Float64Array(size);
    case 'float16':
      return new Uint16Array(size); // bits only; decode on read
    case 'uint8':
      return new Uint8Array(size);
    case 'uint16':
      return new Uint16Array(size);
    case 'uint32':
      return new Uint32Array(size);
    case 'int8':
      return new Int8Array(size);
    case 'int16':
      return new Int16Array(size);
    case 'int32':
      return new Int32Array(size);
    case 'int64':
      return new BigInt64Array(size);
    default: {
      const never: never = dtype;
      throw new Error(`Unsupported dtype: ${String(never)}`);
    }
  }
}

/**
 * Wraps an existing ArrayBuffer as the NumericArray for `dtype` without copying.
 * `byteOffset`/`length` are in bytes/elements respectively; `length` defaults to
 * filling the rest of the buffer.
 */
export function viewNumericArray(
  dtype: DTypeString,
  buffer: ArrayBufferLike,
  byteOffset = 0,
  length?: number,
): NumericArray {
  switch (dtype) {
    case 'float32':
      return new Float32Array(buffer, byteOffset, length);
    case 'float64':
      return new Float64Array(buffer, byteOffset, length);
    case 'float16':
    case 'uint16':
      return new Uint16Array(buffer, byteOffset, length);
    case 'uint8':
      return new Uint8Array(buffer, byteOffset, length);
    case 'uint32':
      return new Uint32Array(buffer, byteOffset, length);
    case 'int8':
      return new Int8Array(buffer, byteOffset, length);
    case 'int16':
      return new Int16Array(buffer, byteOffset, length);
    case 'int32':
      return new Int32Array(buffer, byteOffset, length);
    case 'int64':
      return new BigInt64Array(buffer, byteOffset, length);
    default: {
      const never: never = dtype;
      throw new Error(`Unsupported dtype: ${String(never)}`);
    }
  }
}

import type { DTypeString, NumericArray } from 'ptensor-ts';

// The typed-array union and allocator live in ptensor-ts (as `NumericArray` /
// `createNumericArray`); re-export under the binding's historical names.
export {
  createNumericArray as createTypedArray,
  type NumericArray as TypedArrayType,
} from 'ptensor-ts';

/** Maps a typed array instance back to its dtype string. */
export const getDtypeFromTypedArray = (data: NumericArray): DTypeString => {
  if (data instanceof Float32Array) return 'float32';
  if (data instanceof Float64Array) return 'float64';
  if (data instanceof Uint8Array) return 'uint8';
  if (data instanceof Uint16Array) return 'uint16';
  if (data instanceof Uint32Array) return 'uint32';
  if (data instanceof Int8Array) return 'int8';
  if (data instanceof Int16Array) return 'int16';
  if (data instanceof Int32Array) return 'int32';
  if (data instanceof BigInt64Array) return 'int64';
  throw new Error('Unsupported typed array type');
};

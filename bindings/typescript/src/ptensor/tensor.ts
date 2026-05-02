import {
  p10_from_data,
  p10_from_data_strided,
  p10_destroy,
  p10_get_size,
  p10_get_size_bytes,
  p10_get_dtype,
  p10_get_shape,
  p10_get_stride,
  p10_get_ndim,
  p10_is_empty,
} from './backends/bun/ffi.js';
import { P10Error } from './p10Error.js';
import { DTypeString, dtypeToNumber, numberToDtype } from './dtype.js';
import { TypedArrayType, getDtypeFromTypedArray, createTypedArray } from './typedArray.js';

export type { DTypeString };
export type { TypedArrayType };

export interface Tensor {
  /** Total number of elements. */
  getSize(): bigint;
  /** Total data size in bytes. */
  getSizeBytes(): bigint;
  /** Dimension sizes. */
  getShape(): bigint[];
  /** Per-element strides. */
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

// ------------------------------------------------------------------ //
// Opaque handle helpers
//
// A Ptensor is a void*. It is kept in a BigUint64Array(1) so that:
//   - p10_from_data writes into it (Ptensor* out)
//   - p10_destroy can null it out (Ptensor*)
//   - accessors receive Number(buf[0]) as a raw-pointer int arg
// ------------------------------------------------------------------ //

function readHandle(buf: BigUint64Array): number {
  return Number(buf[0]);
}

function ffiInt(v: unknown): number {
  return v as number;
}

function ffiU64(v: unknown): bigint {
  const n = v as bigint | number;
  return typeof n === 'bigint' ? n : BigInt(n);
}

class TensorImpl implements Tensor {
  private _buf: BigUint64Array; // [0] = opaque Ptensor value
  private _owner?: object;      // keeps JS data buffer alive for view tensors

  constructor(buf: BigUint64Array, owner?: object) {
    this._buf = buf;
    this._owner = owner;
  }

  getSize(): bigint {
    return ffiU64(p10_get_size(readHandle(this._buf)));
  }

  getSizeBytes(): bigint {
    return ffiU64(p10_get_size_bytes(readHandle(this._buf)));
  }

  getNdim(): number {
    return Number(ffiU64(p10_get_ndim(readHandle(this._buf))));
  }

  getShape(): bigint[] {
    const ndim = this.getNdim();
    const shapeBuf = new BigInt64Array(ndim);
    P10Error.check(ffiInt(p10_get_shape(readHandle(this._buf), shapeBuf, ndim)));
    return Array.from(shapeBuf).map(BigInt);
  }

  getStride(): bigint[] {
    const ndim = this.getNdim();
    const strideBuf = new BigInt64Array(ndim);
    P10Error.check(ffiInt(p10_get_stride(readHandle(this._buf), strideBuf, ndim)));
    return Array.from(strideBuf).map(BigInt);
  }

  getDtype(): DTypeString {
    const code = ffiInt(p10_get_dtype(readHandle(this._buf)));
    const dtype = numberToDtype[code];
    if (!dtype) throw new P10Error(0, `Unknown dtype code: ${code}`);
    return dtype;
  }

  isEmpty(): boolean {
    return ffiInt(p10_is_empty(readHandle(this._buf))) !== 0;
  }

  delete(): void {
    if (this._buf[0] !== 0n) {
      P10Error.check(ffiInt(p10_destroy(this._buf)));
    }
    this._owner = undefined;
  }
}

// ------------------------------------------------------------------ //
// Public factory functions
// ------------------------------------------------------------------ //

/**
 * Creates a Tensor view over the given TypedArray.
 * The array must stay in scope for as long as the Tensor is used.
 * Custom per-element strides (not byte strides) may optionally be supplied.
 */
export function fromArray(
  data: TypedArrayType,
  shape: number[],
  strides?: number[]
): Tensor {
  const dtype = getDtypeFromTypedArray(data);
  const dtypeNum = dtypeToNumber[dtype];
  const shapeArr = new BigInt64Array(shape.map(BigInt));
  const buf = new BigUint64Array(1);

  let err: number;
  if (strides) {
    const stridesArr = new BigInt64Array(strides.map(BigInt));
    err = ffiInt(
      p10_from_data_strided(buf, dtypeNum, shapeArr, stridesArr, shape.length, data)
    );
  } else {
    err = ffiInt(p10_from_data(buf, dtypeNum, shapeArr, shape.length, data));
  }

  P10Error.check(err);
  return new TensorImpl(buf, data);
}

/**
 * Creates a Tensor filled with zeros. The backing TypedArray is owned
 * by the returned Tensor and kept alive until delete() is called.
 */
export function zeros(shape: number[], dtype: DTypeString = 'float32'): Tensor {
  const size = shape.reduce((a, b) => a * b, 1);
  const data = createTypedArray(dtype, size);
  return fromArray(data, shape);
}


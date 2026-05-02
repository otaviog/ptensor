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
} from './backends/bun/ffi';
import { P10Error } from './p10Error';
import { DTypeString, dtypeToNumber, numberToDtype } from './dtype';
import { TypedArrayType, getDtypeFromTypedArray, createTypedArray } from './typedArray';
import { ffiInt, ffiU64, readHandle, newHandleBuf } from './_internal';
import type { Tensor } from './types';

export type { DTypeString };
export type { TypedArrayType };
export type { Tensor };

class TensorImpl implements Tensor {
  /** @internal – do not access outside this module or infer.ts */ readonly _buf: BigUint64Array;
  private _owner?: object;

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
  const buf = newHandleBuf();

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

// ------------------------------------------------------------------ //
// Internal helpers (used by infer.ts)
// ------------------------------------------------------------------ //

/** @internal Wraps an existing native Ptensor handle into a Tensor. */
export function _wrapHandle(buf: BigUint64Array): Tensor {
  return new TensorImpl(buf);
}

/** @internal Extracts the raw Ptensor handle value from a Tensor. */
export function _getRawHandle(t: Tensor): bigint {
  return (t as TensorImpl)._buf[0];
}


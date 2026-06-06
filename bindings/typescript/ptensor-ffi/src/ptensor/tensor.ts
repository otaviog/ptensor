import { toArrayBuffer } from 'bun:ffi';
import { type NumericArray, type Tensor as TensorData, viewNumericArray } from 'ptensor-ts';
import { ffiInt, ffiU64, newHandleBuf, readHandle } from './_internal';
import {
  p10_destroy,
  p10_from_data,
  p10_from_data_strided,
  p10_get_data,
  p10_get_dtype,
  p10_get_ndim,
  p10_get_shape,
  p10_get_size,
  p10_get_size_bytes,
  p10_get_stride,
  p10_is_empty,
} from './backends/bun/ffi';
import { type DTypeString, dtypeToNumber, numberToDtype } from './dtype';
import { P10Error } from './p10Error';
import { createTypedArray, getDtypeFromTypedArray, type TypedArrayType } from './typedArray';
import type { PTensor } from './types';

export type { DTypeString, PTensor, TypedArrayType };

class PTensorImpl implements PTensor {
  /** @internal – do not access outside this module. */ readonly _buf: BigUint64Array;
  /** Keeps the JS backing array alive so the C view's pointer stays valid.
   * Write-only on purpose: its existence is the GC root, never read. */
  // biome-ignore lint/correctness/noUnusedPrivateClassMembers: GC root, retained not read
  private _owner?: object;
  readonly shape: number[];
  readonly stride: number[];
  readonly dtype: DTypeString;

  constructor(buf: BigUint64Array, owner?: object) {
    this._buf = buf;
    this._owner = owner;

    const h = readHandle(buf);
    const ndim = Number(ffiU64(p10_get_ndim(h)));

    const shapeBuf = new BigInt64Array(ndim);
    P10Error.check(ffiInt(p10_get_shape(h, shapeBuf, ndim)));
    this.shape = Array.from(shapeBuf, Number);

    const strideBuf = new BigInt64Array(ndim);
    P10Error.check(ffiInt(p10_get_stride(h, strideBuf, ndim)));
    this.stride = Array.from(strideBuf, Number);

    const code = ffiInt(p10_get_dtype(h));
    const dtype = numberToDtype[code];
    if (!dtype) throw new P10Error(0, `Unknown dtype code: ${code}`);
    this.dtype = dtype;
  }

  size(): bigint {
    return ffiU64(p10_get_size(readHandle(this._buf)));
  }

  sizeBytes(): bigint {
    return ffiU64(p10_get_size_bytes(readHandle(this._buf)));
  }

  ndim(): number {
    return this.shape.length;
  }

  isEmpty(): boolean {
    return ffiInt(p10_is_empty(readHandle(this._buf))) !== 0;
  }

  data(): NumericArray {
    const byteLength = Number(this.sizeBytes());
    if (byteLength === 0) {
      return viewNumericArray(this.dtype, new ArrayBuffer(0));
    }
    const ptr = p10_get_data(readHandle(this._buf)) as number;
    if (!ptr) throw new P10Error(0, 'Tensor data pointer is null.');
    const ab = toArrayBuffer(ptr, 0, byteLength);
    return viewNumericArray(this.dtype, ab);
  }

  toTensor(): TensorData {
    return {
      dtype: this.dtype,
      shape: this.shape,
      stride: this.stride,
      // `.slice()` copies out of the native view into a JS-owned typed array.
      data: this.data().slice() as NumericArray,
    };
  }

  destroy(): void {
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
 * Creates a PTensor view over the given TypedArray. The array must stay in
 * scope for as long as the PTensor is used (the PTensor holds a reference to it
 * so the C view's pointer stays valid). Custom per-element strides (not byte
 * strides) may optionally be supplied.
 */
export function fromArray(data: TypedArrayType, shape: number[], strides?: number[]): PTensor {
  const dtype = getDtypeFromTypedArray(data);
  const dtypeNum = dtypeToNumber[dtype];
  const shapeArr = new BigInt64Array(shape.map(BigInt));
  const buf = newHandleBuf();

  let err: number;
  if (strides) {
    const stridesArr = new BigInt64Array(strides.map(BigInt));
    err = ffiInt(p10_from_data_strided(buf, dtypeNum, shapeArr, stridesArr, shape.length, data));
  } else {
    err = ffiInt(p10_from_data(buf, dtypeNum, shapeArr, shape.length, data));
  }

  P10Error.check(err);
  return new PTensorImpl(buf, data);
}

/**
 * Creates a PTensor filled with zeros. The backing TypedArray is owned by the
 * returned PTensor and kept alive until destroy() is called.
 */
export function zeros(shape: number[], dtype: DTypeString = 'float32'): PTensor {
  const size = shape.reduce((a, b) => a * b, 1);
  const data = createTypedArray(dtype, size);
  return fromArray(data, shape);
}

// ------------------------------------------------------------------ //
// Internal helpers (used by media.ts)
// ------------------------------------------------------------------ //

/** @internal Wraps an existing native Ptensor handle into a PTensor. */
export function _wrapHandle(buf: BigUint64Array): PTensor {
  return new PTensorImpl(buf);
}

/** @internal Extracts the raw Ptensor handle value from a PTensor. */
export function _getRawHandle(t: PTensor): bigint {
  return (t as PTensorImpl)._buf[0];
}

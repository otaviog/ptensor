import type { PTensorWasmModule } from './module';
import {
  withPtrOutSlot,
  mallocBuffer,
  mallocI64Array,
  readI64Array,
  readLastErrorMessage,
} from './memory';
import type { Tensor, DTypeString, TypedArrayType } from '../../types';
import { dtypeToNumber, numberToDtype } from '../../dtype';
import { getDtypeFromTypedArray, createTypedArray } from '../../typedArray';

const P10_OK = 0;

function check(mod: PTensorWasmModule, err: number): void {
  if (err !== P10_OK) {
    const msg = readLastErrorMessage(mod) ?? 'Unknown error';
    throw new Error(`P10Error(${err}): ${msg}`);
  }
}

/**
 * WASM implementation of the `Tensor` interface.
 *
 * `_dataPtr` is non-zero for tensors whose backing data was copied to the
 * WASM heap by `fromArray`. It must be freed AFTER calling `_p10_destroy`.
 * For tensors returned from C (e.g. future infer outputs), `_dataPtr` is 0
 * and the data is owned/freed by the C library.
 */
export class WasmTensorImpl implements Tensor {
  private _mod: PTensorWasmModule;
  private _handle: number;
  private _dataPtr: number;

  constructor(mod: PTensorWasmModule, handle: number, dataPtr = 0) {
    this._mod = mod;
    this._handle = handle;
    this._dataPtr = dataPtr;
  }

  /** @internal Raw 32-bit WASM pointer for the Ptensor handle. */
  get rawHandle(): number {
    return this._handle;
  }

  getSize(): bigint {
    return BigInt(this._mod._p10_get_size(this._handle));
  }

  getSizeBytes(): bigint {
    return BigInt(this._mod._p10_get_size_bytes(this._handle));
  }

  getNdim(): number {
    return this._mod._p10_get_ndim(this._handle);
  }

  getShape(): bigint[] {
    const ndim = this.getNdim();
    if (ndim === 0) return [];
    const ptr = this._mod._malloc(ndim * 8);
    check(this._mod, this._mod._p10_get_shape(this._handle, ptr, ndim));
    const result = readI64Array(this._mod, ptr, ndim);
    this._mod._free(ptr);
    return result;
  }

  getStride(): bigint[] {
    const ndim = this.getNdim();
    if (ndim === 0) return [];
    const ptr = this._mod._malloc(ndim * 8);
    check(this._mod, this._mod._p10_get_stride(this._handle, ptr, ndim));
    const result = readI64Array(this._mod, ptr, ndim);
    this._mod._free(ptr);
    return result;
  }

  getDtype(): DTypeString {
    const code = this._mod._p10_get_dtype(this._handle);
    const dtype = numberToDtype[code];
    if (!dtype) throw new Error(`Unknown dtype code: ${code}`);
    return dtype;
  }

  isEmpty(): boolean {
    return this._mod._p10_is_empty(this._handle) !== 0;
  }

  delete(): void {
    if (this._handle !== 0) {
      // p10_destroy takes Ptensor* (a pointer to the handle value).
      // Allocate a 4-byte stack slot, write the current handle into it,
      // then call destroy so the C library can free the tensor and zero the slot.
      const sp = this._mod.stackSave();
      const slot = this._mod.stackAlloc(4);
      this._mod.setValue(slot, this._handle, 'i32');
      this._mod._p10_destroy(slot);
      this._mod.stackRestore(sp);
      this._handle = 0;
    }
    if (this._dataPtr !== 0) {
      this._mod._free(this._dataPtr);
      this._dataPtr = 0;
    }
  }
}

/**
 * Creates a Tensor by copying `data` to the WASM heap and calling
 * `p10_from_data` (or `p10_from_data_strided` when strides are provided).
 *
 * The WASM heap copy is owned by the returned Tensor and freed on `delete()`.
 */
export function fromArray(
  mod: PTensorWasmModule,
  data: TypedArrayType,
  shape: number[],
  strides?: number[]
): Tensor {
  const dtype = getDtypeFromTypedArray(data);
  const dtypeNum = dtypeToNumber[dtype];
  const shapePtr = mallocI64Array(mod, shape.map(BigInt));
  const dataPtr = mallocBuffer(mod, data);

  let err: number;
  let handle: number;

  try {
    if (strides) {
      const stridesPtr = mallocI64Array(mod, strides.map(BigInt));
      try {
        [err, handle] = withPtrOutSlot(mod, (slot) =>
          mod._p10_from_data_strided(slot, dtypeNum, shapePtr, stridesPtr, shape.length, dataPtr)
        );
      } finally {
        mod._free(stridesPtr);
      }
    } else {
      [err, handle] = withPtrOutSlot(mod, (slot) =>
        mod._p10_from_data(slot, dtypeNum, shapePtr, shape.length, dataPtr)
      );
    }
  } finally {
    mod._free(shapePtr);
  }

  try {
    check(mod, err!);
  } catch (e) {
    mod._free(dataPtr);
    throw e;
  }

  return new WasmTensorImpl(mod, handle!, dataPtr);
}

/**
 * Creates a zero-filled Tensor on the WASM heap.
 */
export function zeros(
  mod: PTensorWasmModule,
  shape: number[],
  dtype: DTypeString = 'float32'
): Tensor {
  const size = shape.reduce((a, b) => a * b, 1);
  const data = createTypedArray(dtype, size);
  return fromArray(mod, data, shape);
}

import * as ffi from './ffi';
import { DType, DTYPE_TO_TYPED_ARRAY, getDTypeSize } from './enums';
import { checkError } from './errors';
import * as koffi from 'koffi';

export type TypedArrayLike =
  | Float32Array
  | Float64Array
  | Uint8Array
  | Uint16Array
  | Uint32Array
  | Int8Array
  | Int16Array
  | Int32Array;

/**
 * Tensor class wrapping the Ptensor C API
 */
export class Tensor {
  private handle: any;
  private disposed = false;

  private constructor(handle: any) {
    this.handle = handle;
  }

  /**
   * Create a tensor from data
   * @param data - TypedArray containing the tensor data
   * @param shape - Shape of the tensor
   * @param dtype - Data type (inferred from data if not provided)
   */
  static fromData(
    data: TypedArrayLike | number[],
    shape: number[],
    dtype?: DType
  ): Tensor {
    // Convert plain arrays to typed arrays
    let typedData: TypedArrayLike;
    if (Array.isArray(data)) {
      if (dtype === undefined) {
        dtype = DType.FLOAT32;
      }
      const ArrayConstructor = DTYPE_TO_TYPED_ARRAY[dtype];
      if (!ArrayConstructor) {
        throw new Error(`Cannot create typed array for dtype ${dtype}`);
      }
      typedData = new ArrayConstructor(data);
    } else {
      typedData = data;
      if (dtype === undefined) {
        dtype = inferDType(typedData);
      }
    }

    // Validate shape
    const expectedSize = shape.reduce((a, b) => a * b, 1);
    if (typedData.length !== expectedSize) {
      throw new Error(
        `Data length ${typedData.length} does not match shape ${shape} (expected ${expectedSize})`
      );
    }

    // Prepare shape array (int64_t*)
    const shapeArray = new BigInt64Array(shape.map(BigInt));

    // Get data pointer
    const dataPtr = koffi.as(typedData, koffi.pointer('uint8_t'));

    // Call C API
    const tensorPtr = [null];
    const errorCode = ffi.p10_from_data(
      tensorPtr,
      dtype,
      koffi.as(shapeArray, koffi.pointer('int64_t')),
      shape.length,
      dataPtr
    );

    checkError(errorCode);

    if (!tensorPtr[0]) {
      throw new Error('Failed to create tensor: null pointer returned');
    }

    return new Tensor(tensorPtr[0]);
  }

  /**
   * Create a tensor filled with zeros
   */
  static zeros(shape: number[], dtype: DType = DType.FLOAT32): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    const ArrayConstructor = DTYPE_TO_TYPED_ARRAY[dtype];
    if (!ArrayConstructor) {
      throw new Error(`Cannot create typed array for dtype ${dtype}`);
    }
    const data = new ArrayConstructor(size);
    return Tensor.fromData(data, shape, dtype);
  }

  /**
   * Create a tensor filled with ones
   */
  static ones(shape: number[], dtype: DType = DType.FLOAT32): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    const ArrayConstructor = DTYPE_TO_TYPED_ARRAY[dtype];
    if (!ArrayConstructor) {
      throw new Error(`Cannot create typed array for dtype ${dtype}`);
    }
    const data = new ArrayConstructor(size).fill(1);
    return Tensor.fromData(data, shape, dtype);
  }

  /**
   * Get the data type of the tensor
   */
  get dtype(): DType {
    this.checkDisposed();
    return ffi.p10_get_dtype(this.handle);
  }

  /**
   * Get the shape of the tensor
   */
  get shape(): number[] {
    this.checkDisposed();
    const ndim = this.ndim;
    const shapeArray = new BigInt64Array(ndim);
    const errorCode = ffi.p10_get_shape(
      this.handle,
      koffi.as(shapeArray, koffi.pointer('int64_t')),
      ndim
    );
    checkError(errorCode);
    return Array.from(shapeArray, Number);
  }

  /**
   * Get the number of dimensions
   */
  get ndim(): number {
    this.checkDisposed();
    return Number(ffi.p10_get_dimensions(this.handle));
  }

  /**
   * Get the total number of elements
   */
  get size(): number {
    this.checkDisposed();
    return Number(ffi.p10_get_size(this.handle));
  }

  /**
   * Get tensor data as a TypedArray
   */
  getData<T extends TypedArrayLike = TypedArrayLike>(): T {
    this.checkDisposed();
    const dataPtr = ffi.p10_get_data(this.handle);
    const size = this.size;
    const dtype = this.dtype;

    const ArrayConstructor = DTYPE_TO_TYPED_ARRAY[dtype];
    if (!ArrayConstructor) {
      throw new Error(`Cannot create typed array for dtype ${dtype}`);
    }

    // Create a view of the data
    const buffer = koffi.decode(dataPtr, koffi.array('uint8_t', size * getDTypeSize(dtype)));
    return new ArrayConstructor(buffer.buffer) as T;
  }

  /**
   * Get tensor data as a plain JavaScript array
   */
  toArray(): number[] {
    return Array.from(this.getData());
  }

  /**
   * Get a string representation of the tensor
   */
  toString(): string {
    this.checkDisposed();
    const shape = this.shape;
    const dtype = DType[this.dtype];
    const size = this.size;
    return `Tensor(shape=[${shape.join(', ')}], dtype=${dtype}, size=${size})`;
  }

  /**
   * Dispose the tensor and free native memory
   */
  dispose(): void {
    if (!this.disposed && this.handle) {
      const handlePtr = [this.handle];
      const errorCode = ffi.p10_destroy(koffi.as(handlePtr, koffi.pointer(ffi.Ptensor)));
      checkError(errorCode);
      this.handle = null;
      this.disposed = true;
    }
  }

  /**
   * Check if tensor has been disposed
   */
  private checkDisposed(): void {
    if (this.disposed) {
      throw new Error('Tensor has been disposed');
    }
  }

  /**
   * Implement Symbol.dispose for automatic cleanup with using keyword
   */
  [Symbol.dispose](): void {
    this.dispose();
  }
}

/**
 * Infer DType from TypedArray
 */
function inferDType(data: TypedArrayLike): DType {
  if (data instanceof Float32Array) return DType.FLOAT32;
  if (data instanceof Float64Array) return DType.FLOAT64;
  if (data instanceof Uint8Array) return DType.UINT8;
  if (data instanceof Uint16Array) return DType.UINT16;
  if (data instanceof Uint32Array) return DType.UINT32;
  if (data instanceof Int8Array) return DType.INT8;
  if (data instanceof Int16Array) return DType.INT16;
  if (data instanceof Int32Array) return DType.INT32;
  throw new Error('Cannot infer dtype from data');
}

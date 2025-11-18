/**
 * PTensor TypeScript Wrapper
 * Provides a user-friendly TypeScript API over the raw WebAssembly bindings
 */

import { createWasmDtype, DTypeString, parseDtype } from './dtype.js';
import { MODULE, P10 } from './module-init.js';
import { getDtypeFromTypedArray, TypedArrayType } from './typed-array.js';

/**
 * Tensor class that wraps the WebAssembly Tensor
 */
export class Tensor {
  private wasmTensor: any;

  private constructor(wasmTensor: any) {
    this.wasmTensor = wasmTensor;
  }

  /**
   * Create a tensor from a typed array
   */
  static fromTypedArray(
    data: TypedArrayType,
    shape: number[]
  ): Tensor {
    const dtype = getDtypeFromTypedArray(data);
    const wasmShape = createWasmShape(shape);
    const wasmDtype = createWasmDtype(dtype);

    // Allocate memory in WebAssembly
    const byteLength = data.byteLength;
    const dataPtr = MODULE._malloc(byteLength);

    try {
      // Copy data to WebAssembly memory
      const heapView = getHeapView(MODULE, dtype, dataPtr, data.length);
      heapView.set(data as any);

      // Create tensor
      const wasmTensor = new MODULE.Tensor();
      const error = wasmTensor.fromData(wasmShape, wasmDtype, dataPtr);

      if (error.isError()) {
        wasmTensor.delete();
        throw new Error(`Failed to create tensor: ${error.toString()}`);
      }
      error.delete();

      const tensor = new Tensor(wasmTensor);

      return tensor;
    } finally {
      MODULE._free(dataPtr);
      wasmShape.delete();
      wasmDtype.delete();
    }
  }

  /**
   * Create a tensor filled with zeros
   */
  static zeros(shape: number[], dtype: DTypeString = 'float32'): Tensor {
    const wasmTensor = MODULE.Tensor.zeros(createWasmShape(shape),
      createWasmDtype(dtype));
    return new Tensor(wasmTensor);
  }

  /**
   * Get tensor shape
   */
  get shape(): number[] {
    return this.wasmTensor.getShape().toArray();
  }

  /**
   * Get tensor dtype
   */
  get dtype(): DTypeString {
    const dtypeStr = this.wasmTensor.getDtypeStr();
    return parseDtype(dtypeStr);
  }

  /**
   * Get total number of elements
   */
  get size(): number {
    return this.wasmTensor.getSize();
  }

  /**
   * Delete the tensor and free memory
   */
  delete(): void {
    if (this.wasmTensor) {
      this.wasmTensor.delete();
      this.wasmTensor = null;
    }
  }
}

const getHeapView = (
  module: P10,
  dtype: DTypeString,
  ptr: number,
  length: number
): TypedArrayType => {
  const byteOffset = ptr;
  // Get the memory buffer - Emscripten exposes it via wasmMemory.buffer
  const buffer = (module as any).wasmMemory?.buffer || (module as any).HEAP8?.buffer;
  if (!buffer) {
    // Try accessing directly from the module
    const anyModule = module as any;
    console.error('Available module properties:', Object.keys(anyModule).filter(k => k.includes('HEAP') || k.includes('memory')).join(', '));
    throw new Error('WebAssembly memory not available');
  }

  switch (dtype) {
    case 'float32': return new Float32Array(buffer, byteOffset, length);
    case 'float64': return new Float64Array(buffer, byteOffset, length);
    case 'uint8': return new Uint8Array(buffer, byteOffset, length);
    case 'uint16': return new Uint16Array(buffer, byteOffset, length);
    case 'uint32': return new Uint32Array(buffer, byteOffset, length);
    case 'int8': return new Int8Array(buffer, byteOffset, length);
    case 'int16': return new Int16Array(buffer, byteOffset, length);
    case 'int32': return new Int32Array(buffer, byteOffset, length);
    default: throw new Error(`Unsupported dtype for heap view: ${dtype}`);
  }
}

const createWasmShape = (shape: number[]): any => {
  return (MODULE.Shape as any).fromArray(shape);
}

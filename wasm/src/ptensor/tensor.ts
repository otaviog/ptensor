/**
 * PTensor TypeScript Wrapper
 * Provides a user-friendly TypeScript API over the raw WebAssembly bindings
 */


import { createDtype, Dtype, DTypeString } from './dtype';
import { MODULE, P10 } from './module-init';
import { createShape, Shape } from './shape';
import { getDtypeFromTypedArray, TypedArrayType } from './typed-array';

export interface Tensor {
  getSize(): number;
  getShape(): Shape;
  getDtype(): Dtype;
  delete(): void;
}

/**
 * Create a tensor from a typed array
 */
export const fromArray = (
  data: TypedArrayType,
  shape: number[]
): Tensor => {
  const dtype = getDtypeFromTypedArray(data);
  const wasmShape = createShape(shape);
  const wasmDtype = createDtype(dtype);

  // Allocate memory in WebAssembly
  const byteLength = data.byteLength;
  const dataPtr = MODULE._malloc(byteLength);

  try {
    // Copy data to WebAssembly memory
    const heapView = getHeapView(MODULE, dtype, dataPtr, data.length);
    heapView.set(data as any);

    // Create tensor
    const tensor = MODULE.Tensor.fromData(wasmShape, wasmDtype, dataPtr);

    if (tensor === null) {
      throw new Error(`Failed to create tensor`);
    }

    return tensor;
  } finally {
    wasmShape.delete();
    wasmDtype.delete();
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


/**
 * Create a tensor filled with zeros
 */
export const zeros = (shape: number[], dtype: DTypeString = 'float32'): Tensor => {
  return MODULE.Tensor.zeros(createShape(shape),
    createDtype(dtype));
}


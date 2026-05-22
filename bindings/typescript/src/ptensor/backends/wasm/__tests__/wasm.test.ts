/**
 * Unit tests for the WASM backend.
 *
 * All tests use a lightweight mock of the Emscripten module so no actual
 * .wasm binary is required.  The mock simulates a minimal in-process tensor
 * store (a Map<number, …>) so that the TypeScript logic can be exercised
 * without building the C library.
 */

import { beforeEach, describe, expect, it } from 'bun:test';
import { loadWasm } from '../index';
import type { PTensorWasmModule } from '../module';
import { fromArray, type WasmTensorImpl, zeros } from '../tensor';

// ------------------------------------------------------------------ //
// Minimal mock of the Emscripten runtime
// ------------------------------------------------------------------ //

interface MockTensor {
  dtype: number;
  shape: bigint[];
  size: bigint;
  sizeBytes: bigint;
  strides: bigint[];
}

function createMockModule(): PTensorWasmModule {
  /** In-process tensor store keyed by 32-bit "pointer" (counter starting at 1). */
  const tensors = new Map<number, MockTensor>();
  let nextHandle = 1;

  /** Simulated WASM linear memory (8 MB). */
  const memory = new ArrayBuffer(8 * 1024 * 1024);
  const heapu8 = new Uint8Array(memory);
  const memView = new DataView(memory);

  // Stack region: 0 – 4095 (grows upward for simplicity in mock).
  // Heap region: starts at 4096.
  let stackPointer = 0;
  let heapTop = 4096;

  function alignTo8(n: number): number {
    return (n + 7) & ~7;
  }

  function malloc(size: number): number {
    const ptr = heapTop;
    heapTop = alignTo8(heapTop + size);
    return ptr;
  }

  function free(_ptr: number): void {
    // No-op in this simple mock.
  }

  function readI64At(ptr: number, index: number): bigint {
    const offset = ptr + index * 8;
    const lo = memView.getUint32(offset, true);
    const hi = memView.getUint32(offset + 4, true);
    return (BigInt(hi) << 32n) | BigInt(lo);
  }

  function writeI64At(ptr: number, index: number, value: bigint): void {
    const offset = ptr + index * 8;
    memView.setUint32(offset, Number(value & 0xffffffffn), true);
    memView.setUint32(offset + 4, Number((value >> 32n) & 0xffffffffn), true);
  }

  return {
    // ---- Stack helpers ----
    stackSave: () => stackPointer,
    stackRestore: (sp) => {
      stackPointer = sp;
    },
    stackAlloc: (bytes) => {
      const ptr = stackPointer;
      stackPointer += alignTo8(bytes);
      return ptr;
    },

    // ---- Heap R/W ----
    getValue: (ptr, type) => {
      if (type === 'i32') return memView.getInt32(ptr, true);
      if (type === 'i64') return Number(readI64At(ptr, 0));
      return 0;
    },
    setValue: (ptr, value, type) => {
      if (type === 'i32') memView.setInt32(ptr, value, true);
    },
    UTF8ToString: (ptr) => {
      let str = '';
      let i = ptr;
      while (heapu8[i] !== 0) str += String.fromCharCode(heapu8[i++]);
      return str;
    },
    HEAPU8: heapu8,
    _malloc: malloc,
    _free: free,

    // ---- Error ----
    _p10_get_last_error_message: () => 0, // no error message in mock

    // ---- Dtype ----
    _p10_dtype_to_string: () => 0,
    _p10_dtype_size_bytes: (dtype) => [4, 8, 2, 1, 2, 4, 1, 2, 4, 8][dtype] ?? 0,

    // ---- Tensor lifecycle ----
    _p10_from_data: (outSlot, dtype, shapePtr, numDims, _dataPtr) => {
      const shape: bigint[] = [];
      for (let i = 0; i < numDims; i++) shape.push(readI64At(shapePtr, i));
      const size = shape.reduce((a, b) => a * b, 1n);
      const elemBytes = [4, 8, 2, 1, 2, 4, 1, 2, 4, 8][dtype] ?? 4;

      // Compute default row-major strides.
      const strides: bigint[] = new Array(numDims).fill(1n);
      for (let i = numDims - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
      }

      const handle = nextHandle++;
      tensors.set(handle, { dtype, shape, size, sizeBytes: size * BigInt(elemBytes), strides });
      memView.setInt32(outSlot, handle, true);
      return 0; // P10_OK
    },

    _p10_from_data_strided: (outSlot, dtype, shapePtr, stridesPtr, numDims, _dataPtr) => {
      const shape: bigint[] = [];
      const strides: bigint[] = [];
      for (let i = 0; i < numDims; i++) {
        shape.push(readI64At(shapePtr, i));
        strides.push(readI64At(stridesPtr, i));
      }
      const size = shape.reduce((a, b) => a * b, 1n);
      const elemBytes = [4, 8, 2, 1, 2, 4, 1, 2, 4, 8][dtype] ?? 4;

      const handle = nextHandle++;
      tensors.set(handle, { dtype, shape, size, sizeBytes: size * BigInt(elemBytes), strides });
      memView.setInt32(outSlot, handle, true);
      return 0;
    },

    _p10_destroy: (tensorPtr) => {
      const handle = memView.getInt32(tensorPtr, true);
      tensors.delete(handle);
      memView.setInt32(tensorPtr, 0, true);
      return 0;
    },

    // ---- Accessors ----
    _p10_get_size: (h) => Number(tensors.get(h)?.size ?? 0n),
    _p10_get_size_bytes: (h) => Number(tensors.get(h)?.sizeBytes ?? 0n),
    _p10_get_dtype: (h) => tensors.get(h)?.dtype ?? 0,
    _p10_get_shape: (h, outPtr, numDims) => {
      const t = tensors.get(h);
      if (!t) return 3; // P10_INVALID_ARGUMENT
      for (let i = 0; i < numDims; i++) writeI64At(outPtr, i, t.shape[i] ?? 0n);
      return 0;
    },
    _p10_get_stride: (h, outPtr, numDims) => {
      const t = tensors.get(h);
      if (!t) return 3;
      for (let i = 0; i < numDims; i++) writeI64At(outPtr, i, t.strides[i] ?? 0n);
      return 0;
    },
    _p10_get_ndim: (h) => tensors.get(h)?.shape.length ?? 0,
    _p10_get_data: () => 0,
    _p10_is_empty: (h) => (tensors.get(h)?.size === 0n ? 1 : 0),

    // Expose tensor store for assertions in tests.
    _tensors: tensors,
  } as PTensorWasmModule & { _tensors: Map<number, MockTensor> };
}

// ------------------------------------------------------------------ //
// Tests
// ------------------------------------------------------------------ //

describe('WASM backend — fromArray', () => {
  let mod: ReturnType<typeof createMockModule>;

  beforeEach(() => {
    mod = createMockModule();
  });

  it('creates a float32 tensor with correct metadata', () => {
    const t = fromArray(mod, new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    expect(t.getDtype()).toBe('float32');
    expect(t.getShape()).toEqual([2n, 3n]);
    expect(t.getNdim()).toBe(2);
    expect(t.getSize()).toBe(6n);
    expect(t.getSizeBytes()).toBe(24n); // 6 × 4
    expect(t.isEmpty()).toBe(false);
    t.delete();
  });

  it('creates a uint8 tensor with correct sizeBytes', () => {
    const t = fromArray(mod, new Uint8Array([0, 128, 255]), [3]);
    expect(t.getDtype()).toBe('uint8');
    expect(t.getSizeBytes()).toBe(3n);
    t.delete();
  });

  it('creates a float64 tensor with correct sizeBytes', () => {
    const t = fromArray(mod, new Float64Array([1.0, 2.0]), [2]);
    expect(t.getDtype()).toBe('float64');
    expect(t.getSizeBytes()).toBe(16n); // 2 × 8
    t.delete();
  });

  it('creates a int64 tensor', () => {
    const t = fromArray(mod, new BigInt64Array([10n, 20n]), [2]);
    expect(t.getDtype()).toBe('int64');
    expect(t.getSizeBytes()).toBe(16n); // 2 × 8
    t.delete();
  });

  it('preserves custom strides', () => {
    const t = fromArray(mod, new Float32Array(12), [2, 3], [6, 2]);
    expect(t.getShape()).toEqual([2n, 3n]);
    expect(t.getStride()).toEqual([6n, 2n]);
    t.delete();
  });

  it('reports isEmpty = true for a zero-element tensor', () => {
    const t = fromArray(mod, new Float32Array(0), [0]);
    expect(t.isEmpty()).toBe(true);
    expect(t.getSize()).toBe(0n);
    t.delete();
  });

  it('computes default row-major strides for [2, 3, 4]', () => {
    const t = fromArray(mod, new Float32Array(24), [2, 3, 4]);
    expect(t.getStride()).toEqual([12n, 4n, 1n]);
    t.delete();
  });
});

describe('WASM backend — zeros', () => {
  let mod: ReturnType<typeof createMockModule>;

  beforeEach(() => {
    mod = createMockModule();
  });

  it("zeros([3]) defaults to 'float32'", () => {
    const t = zeros(mod, [3]);
    expect(t.getDtype()).toBe('float32');
    expect(t.getShape()).toEqual([3n]);
    t.delete();
  });

  it("zeros([2, 2], 'int32') uses int32 dtype", () => {
    const t = zeros(mod, [2, 2], 'int32');
    expect(t.getDtype()).toBe('int32');
    expect(t.getSizeBytes()).toBe(16n); // 4 × 4
    t.delete();
  });
});

describe('WASM backend — delete', () => {
  let mod: ReturnType<typeof createMockModule> & { _tensors: Map<number, MockTensor> };

  beforeEach(() => {
    mod = createMockModule() as ReturnType<typeof createMockModule> & {
      _tensors: Map<number, MockTensor>;
    };
  });

  it('frees the handle and data pointer on delete()', () => {
    const t = fromArray(mod, new Float32Array([1, 2]), [2]);
    const impl = t as WasmTensorImpl;
    expect(impl.rawHandle).not.toBe(0);

    t.delete();

    expect(impl.rawHandle).toBe(0);
    // After delete the tensor store should be empty.
    expect(mod._tensors.size).toBe(0);
  });

  it('double-delete is safe (handle is 0 after first delete)', () => {
    const t = fromArray(mod, new Float32Array([1]), [1]);
    t.delete();
    expect(() => t.delete()).not.toThrow();
  });
});

describe('WASM backend — loadWasm()', () => {
  it('accepts a pre-instantiated module and returns the API', async () => {
    const mod = createMockModule();
    const api = await loadWasm(mod);

    const t = api.fromArray(new Float32Array([1, 2, 3]), [3]);
    expect(t.getDtype()).toBe('float32');
    expect(t.getShape()).toEqual([3n]);
    t.delete();
  });

  it('api.zeros() works through loadWasm()', async () => {
    const api = await loadWasm(createMockModule());
    const t = api.zeros([4, 4]);
    expect(t.getShape()).toEqual([4n, 4n]);
    expect(t.getDtype()).toBe('float32');
    t.delete();
  });
});

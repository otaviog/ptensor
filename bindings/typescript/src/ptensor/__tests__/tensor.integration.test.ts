import { describe, it, expect, afterEach } from 'bun:test';
import { fromArray, zeros, type Tensor } from '../tensor.js';

// ------------------------------------------------------------------ //
// These tests require the native libptensor_capi library to be present.
// The preload script (src/__tests__/setup.ts) sets PTENSOR_LIB_PATH
// automatically when running from the build tree.
// ------------------------------------------------------------------ //

describe('Tensor integration (real C library)', () => {
  const tensors: Tensor[] = [];

  function track(t: Tensor): Tensor {
    tensors.push(t);
    return t;
  }

  afterEach(() => {
    for (const t of tensors.splice(0)) {
      t.delete();
    }
  });

  // ---------------------------------------------------------------- //
  // fromArray — basic creation and metadata
  // ---------------------------------------------------------------- //

  it('float32: correct dtype, shape, ndim, size, sizeBytes', () => {
    const t = track(fromArray(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]));
    expect(t.getDtype()).toBe('float32');
    expect(t.getShape()).toEqual([2n, 3n]);
    expect(t.getNdim()).toBe(2);
    expect(t.getSize()).toBe(6n);
    expect(t.getSizeBytes()).toBe(24n); // 6 * 4
  });

  it('float64: correct dtype and sizeBytes', () => {
    const t = track(fromArray(new Float64Array([1.0, 2.0]), [2]));
    expect(t.getDtype()).toBe('float64');
    expect(t.getSizeBytes()).toBe(16n); // 2 * 8
  });

  it('int32: correct dtype and sizeBytes', () => {
    const t = track(fromArray(new Int32Array([10, 20, 30]), [3]));
    expect(t.getDtype()).toBe('int32');
    expect(t.getSizeBytes()).toBe(12n); // 3 * 4
  });

  it('uint8: correct dtype and sizeBytes', () => {
    const t = track(fromArray(new Uint8Array([0, 128, 255]), [3]));
    expect(t.getDtype()).toBe('uint8');
    expect(t.getSizeBytes()).toBe(3n);
  });

  it('int64: correct dtype and sizeBytes', () => {
    const t = track(fromArray(new BigInt64Array([100n, 200n]), [2]));
    expect(t.getDtype()).toBe('int64');
    expect(t.getSizeBytes()).toBe(16n); // 2 * 8
  });

  it('int8, int16, uint16, uint32: correct dtypes', () => {
    const cases = [
      [new Int8Array(2),   'int8'],
      [new Int16Array(2),  'int16'],
      [new Uint16Array(2), 'uint16'],
      [new Uint32Array(2), 'uint32'],
    ] as const;
    for (const [data, dtype] of cases) {
      const t = track(fromArray(data, [2]));
      expect(t.getDtype()).toBe(dtype);
    }
  });

  // ---------------------------------------------------------------- //
  // Shape and strides
  // ---------------------------------------------------------------- //

  it('3D tensor: shape, ndim, size', () => {
    const t = track(fromArray(new Float32Array(24), [2, 3, 4]));
    expect(t.getShape()).toEqual([2n, 3n, 4n]);
    expect(t.getNdim()).toBe(3);
    expect(t.getSize()).toBe(24n);
  });

  it('row-major strides for shape [2, 3]', () => {
    const t = track(fromArray(new Float32Array(6), [2, 3]));
    expect(t.getStride()).toEqual([3n, 1n]);
  });

  it('row-major strides for shape [2, 3, 4]', () => {
    const t = track(fromArray(new Float32Array(24), [2, 3, 4]));
    expect(t.getStride()).toEqual([12n, 4n, 1n]);
  });

  it('custom strides are preserved', () => {
    const t = track(fromArray(new Float32Array(12), [2, 3], [6, 2]));
    expect(t.getShape()).toEqual([2n, 3n]);
    expect(t.getStride()).toEqual([6n, 2n]);
  });

  // ---------------------------------------------------------------- //
  // isEmpty
  // ---------------------------------------------------------------- //

  it('non-empty tensor is not empty', () => {
    const t = track(fromArray(new Float32Array([1, 2, 3]), [3]));
    expect(t.isEmpty()).toBe(false);
  });

  it('zero-element tensor is empty', () => {
    const t = track(fromArray(new Float32Array(0), [0]));
    expect(t.isEmpty()).toBe(true);
    expect(t.getSize()).toBe(0n);
  });

  // ---------------------------------------------------------------- //
  // zeros
  // ---------------------------------------------------------------- //

  it('zeros([4, 4]) — float32 by default', () => {
    const t = track(zeros([4, 4]));
    expect(t.getDtype()).toBe('float32');
    expect(t.getShape()).toEqual([4n, 4n]);
    expect(t.getSize()).toBe(16n);
    expect(t.isEmpty()).toBe(false);
  });

  it("zeros([3], 'int32') — int32 dtype", () => {
    const t = track(zeros([3], 'int32'));
    expect(t.getDtype()).toBe('int32');
    expect(t.getSizeBytes()).toBe(12n);
  });

  it("zeros([2, 2], 'float64') — float64 dtype", () => {
    const t = track(zeros([2, 2], 'float64'));
    expect(t.getDtype()).toBe('float64');
    expect(t.getSizeBytes()).toBe(32n); // 4 * 8
  });

  // ---------------------------------------------------------------- //
  // delete
  // ---------------------------------------------------------------- //

  it('delete() can be called safely', () => {
    // Not tracked; we call delete manually.
    const t = fromArray(new Float32Array([1, 2]), [2]);
    expect(() => t.delete()).not.toThrow();
  });
});

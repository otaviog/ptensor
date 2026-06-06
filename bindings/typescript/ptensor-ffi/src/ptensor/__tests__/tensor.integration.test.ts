import { afterEach, describe, expect, it } from 'bun:test';
import { fromArray, type PTensor, zeros } from '../tensor';

// ------------------------------------------------------------------ //
// These tests require the native libptensor_capi library to be present.
// The preload script (src/__tests__/setup.ts) sets PTENSOR_LIB_PATH
// automatically when running from the build tree.
// ------------------------------------------------------------------ //

describe('PTensor integration (real C library)', () => {
  const tensors: PTensor[] = [];

  function track(t: PTensor): PTensor {
    tensors.push(t);
    return t;
  }

  afterEach(() => {
    for (const t of tensors.splice(0)) {
      t.destroy();
    }
  });

  // ---------------------------------------------------------------- //
  // fromArray — basic creation and metadata
  // ---------------------------------------------------------------- //

  it('float32: correct dtype, shape, ndim, size, sizeBytes', () => {
    const t = track(fromArray(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]));
    expect(t.dtype).toBe('float32');
    expect(t.shape).toEqual([2, 3]);
    expect(t.ndim()).toBe(2);
    expect(t.size()).toBe(6n);
    expect(t.sizeBytes()).toBe(24n); // 6 * 4
  });

  it('float64: correct dtype and sizeBytes', () => {
    const t = track(fromArray(new Float64Array([1.0, 2.0]), [2]));
    expect(t.dtype).toBe('float64');
    expect(t.sizeBytes()).toBe(16n); // 2 * 8
  });

  it('int32: correct dtype and sizeBytes', () => {
    const t = track(fromArray(new Int32Array([10, 20, 30]), [3]));
    expect(t.dtype).toBe('int32');
    expect(t.sizeBytes()).toBe(12n); // 3 * 4
  });

  it('uint8: correct dtype and sizeBytes', () => {
    const t = track(fromArray(new Uint8Array([0, 128, 255]), [3]));
    expect(t.dtype).toBe('uint8');
    expect(t.sizeBytes()).toBe(3n);
  });

  it('int64: correct dtype and sizeBytes', () => {
    const t = track(fromArray(new BigInt64Array([100n, 200n]), [2]));
    expect(t.dtype).toBe('int64');
    expect(t.sizeBytes()).toBe(16n); // 2 * 8
  });

  it('int8, int16, uint16, uint32: correct dtypes', () => {
    const cases = [
      [new Int8Array(2), 'int8'],
      [new Int16Array(2), 'int16'],
      [new Uint16Array(2), 'uint16'],
      [new Uint32Array(2), 'uint32'],
    ] as const;
    for (const [data, dtype] of cases) {
      const t = track(fromArray(data, [2]));
      expect(t.dtype).toBe(dtype);
    }
  });

  // ---------------------------------------------------------------- //
  // Shape and strides
  // ---------------------------------------------------------------- //

  it('3D tensor: shape, ndim, size', () => {
    const t = track(fromArray(new Float32Array(24), [2, 3, 4]));
    expect(t.shape).toEqual([2, 3, 4]);
    expect(t.ndim()).toBe(3);
    expect(t.size()).toBe(24n);
  });

  it('row-major strides for shape [2, 3]', () => {
    const t = track(fromArray(new Float32Array(6), [2, 3]));
    expect(t.stride).toEqual([3, 1]);
  });

  it('row-major strides for shape [2, 3, 4]', () => {
    const t = track(fromArray(new Float32Array(24), [2, 3, 4]));
    expect(t.stride).toEqual([12, 4, 1]);
  });

  it('custom strides are preserved', () => {
    const t = track(fromArray(new Float32Array(12), [2, 3], [6, 2]));
    expect(t.shape).toEqual([2, 3]);
    expect(t.stride).toEqual([6, 2]);
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
    expect(t.size()).toBe(0n);
  });

  // ---------------------------------------------------------------- //
  // zeros
  // ---------------------------------------------------------------- //

  it('zeros([4, 4]) — float32 by default', () => {
    const t = track(zeros([4, 4]));
    expect(t.dtype).toBe('float32');
    expect(t.shape).toEqual([4, 4]);
    expect(t.size()).toBe(16n);
    expect(t.isEmpty()).toBe(false);
  });

  it("zeros([3], 'int32') — int32 dtype", () => {
    const t = track(zeros([3], 'int32'));
    expect(t.dtype).toBe('int32');
    expect(t.sizeBytes()).toBe(12n);
  });

  it("zeros([2, 2], 'float64') — float64 dtype", () => {
    const t = track(zeros([2, 2], 'float64'));
    expect(t.dtype).toBe('float64');
    expect(t.sizeBytes()).toBe(32n); // 4 * 8
  });

  // ---------------------------------------------------------------- //
  // delete
  // ---------------------------------------------------------------- //

  it('delete() can be called safely', () => {
    // Not tracked; we call delete manually.
    const t = fromArray(new Float32Array([1, 2]), [2]);
    expect(() => t.destroy()).not.toThrow();
  });
});

import { describe, expect, it } from 'bun:test';
import { dtypeSizeBytes } from '../dtype';
import { parseTensorJson } from '../tensorJson';
import {
  contiguousStride,
  numElements,
  parse,
  tensorFromJson,
  tensorToJson,
} from '../tensor';

function makeJson(dtype: string, shape: number[], data: ArrayBufferView): string {
  const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const blob = Buffer.from(bytes).toString('base64');
  return JSON.stringify({ dtype, shape, stride: contiguousStride(shape), blob });
}

describe('ptensor-ts', () => {
  it('round-trips a float32 tensor through JSON', () => {
    const src = new Float32Array([1, 2, 3, 4, 5, 6]);
    const raw = makeJson('float32', [2, 3], src);
    const t = parse(raw);
    expect(t.dtype).toBe('float32');
    expect(t.shape).toEqual([2, 3]);
    expect(Array.from(t.data as Float32Array)).toEqual([1, 2, 3, 4, 5, 6]);

    const back = tensorToJson(t);
    expect(parseTensorJson(JSON.stringify(back)).blob).toBe(JSON.parse(raw).blob);
  });

  it('decodes int64 into BigInt64Array', () => {
    const src = new BigInt64Array([1n, -2n, 3n]);
    const t = tensorFromJson(JSON.parse(makeJson('int64', [3], src)));
    expect(t.data).toBeInstanceOf(BigInt64Array);
    expect(Array.from(t.data as BigInt64Array)).toEqual([1n, -2n, 3n]);
  });

  it('unwraps an LLDB-style summary prefix', () => {
    const src = new Uint8Array([10, 20, 30]);
    const inner = makeJson('uint8', [3], src);
    const t = parse(`(const char *) $7 = 0x0000000100 "${inner.replace(/"/g, '\\"')}"`);
    expect(Array.from(t.data as Uint8Array)).toEqual([10, 20, 30]);
  });

  it('helpers compute strides and sizes', () => {
    expect(contiguousStride([2, 3, 4])).toEqual([12, 4, 1]);
    expect(numElements([2, 3, 4])).toBe(24);
    expect(dtypeSizeBytes.int64).toBe(8);
  });

  it('rejects an unknown dtype', () => {
    expect(() => tensorFromJson({ dtype: 'bogus', shape: [1], stride: [1], blob: '' })).toThrow();
  });
});

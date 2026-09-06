import { describe, expect, it } from 'bun:test';
import { dtypeSizeBytes } from '../dtype';
import { P10Error } from '../p10error';
import {
  decodeTensorBlob,
  parse,
  parseTensorJson,
  type TensorJson,
  tensorFromJson,
  tensorToJson,
  validateTensorJson,
} from '../tensorJson';
import { contiguousStride, numElements } from '../tensor';

function makeJsonObject(
  dtype: string,
  shape: number[],
  data: ArrayBufferView,
): TensorJson {
  const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  return {
    dtype,
    shape,
    stride: contiguousStride(shape),
    size_bytes: bytes.byteLength,
    encoding: 'base64',
    blob: Buffer.from(bytes).toString('base64'),
  };
}

function makeJson(dtype: string, shape: number[], data: ArrayBufferView): string {
  return JSON.stringify(makeJsonObject(dtype, shape, data));
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
    expect(back.encoding).toBe('base64');
    expect(back.size_bytes).toBe(24);
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
    expect(() =>
      tensorFromJson({
        dtype: 'bogus',
        shape: [1],
        stride: [1],
        size_bytes: 0,
        encoding: 'base64',
        blob: '',
      }),
    ).toThrow(P10Error);
  });

  it('decodes a zstd-compressed blob', () => {
    const data = new Float32Array([1.5, -2.5, 3.5, 4.5]);
    const raw = new Uint8Array(data.buffer);
    const json = makeJsonObject('float32', [4], data);
    // `size_bytes` stays the uncompressed count, as the C++ encoder writes it.
    json.encoding = 'base64+zstd';
    json.blob = Buffer.from(Bun.zstdCompressSync(raw)).toString('base64');

    const tensor = tensorFromJson(json);
    expect(Array.from(tensor.data)).toEqual([1.5, -2.5, 3.5, 4.5]);
  });

  it('widens a float16 payload to Float32Array', () => {
    // 1.0, -2.0, 0.5 as IEEE 754 half.
    const halves = new Uint16Array([0x3c00, 0xc000, 0x3800]);
    const json = makeJsonObject('float16', [3], halves);

    const tensor = tensorFromJson(json);
    expect(tensor.dtype).toBe('float16');
    expect(tensor.data).toBeInstanceOf(Float32Array);
    expect(Array.from(tensor.data)).toEqual([1, -2, 0.5]);
  });

  it('rejects a zstd-compressed blob that does not decompress', () => {
    const json = { ...makeJsonObject('uint8', [3], new Uint8Array([1, 2, 3])) };
    json.encoding = 'base64+zstd';
    expect(() => tensorFromJson(json)).toThrow(/Could not zstd-decompress/);
  });

  it('exposes the raw bytes of either encoding', () => {
    const raw = new Uint8Array([9, 8, 7, 6]);
    const plain = makeJsonObject('uint8', [4], raw);
    expect(Array.from(decodeTensorBlob(plain))).toEqual([9, 8, 7, 6]);

    const compressed = { ...plain };
    compressed.encoding = 'base64+zstd';
    compressed.blob = Buffer.from(Bun.zstdCompressSync(raw)).toString('base64');
    expect(Array.from(decodeTensorBlob(compressed))).toEqual([9, 8, 7, 6]);
  });

  it("rejects a blob whose length disagrees with 'size_bytes'", () => {
    const json = { ...makeJsonObject('uint8', [3], new Uint8Array([1, 2, 3])) };
    json.size_bytes = 4;
    expect(() => tensorFromJson(json)).toThrow(/'size_bytes' says 4/);
  });

  it('rejects a blob that is not a whole number of elements', () => {
    const json = makeJsonObject('float32', [1], new Uint8Array([1, 2, 3]));
    expect(() => tensorFromJson(json)).toThrow(/not a multiple of 4/);
  });

  it('reports a missing JSON object and malformed JSON', () => {
    expect(() => parseTensorJson('no object here')).toThrow(/Could not find a JSON object/);
    expect(() => parseTensorJson('{not json}')).toThrow(/Could not parse tensor JSON/);
  });

  it('coerces stringified shape and stride numbers', () => {
    const json = parseTensorJson(
      '{"dtype":"uint8","shape":["2","2"],"stride":["2","1"],"size_bytes":4,"encoding":"base64","blob":"AQIDBA=="}',
    );
    expect(json.shape).toEqual([2, 2]);
    expect(json.stride).toEqual([2, 1]);
  });
});

describe('validateTensorJson', () => {
  const valid: TensorJson = {
    dtype: 'float32',
    shape: [2, 3],
    stride: [3, 1],
    size_bytes: 24,
    encoding: 'base64',
    blob: 'AAAAAA==',
  };

  it('accepts a well-formed object and returns the same reference', () => {
    const result = validateTensorJson(valid);
    expect(result).toBe(valid);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([2, 3]);
    expect(result.stride).toEqual([3, 1]);
    expect(result.size_bytes).toBe(24);
    expect(result.encoding).toBe('base64');
    expect(result.blob).toBe('AAAAAA==');
  });

  it('accepts the compressed encoding', () => {
    expect(validateTensorJson({ ...valid, encoding: 'base64+zstd' }).encoding).toBe('base64+zstd');
  });

  it('keeps extra fields untouched', () => {
    const extra = { ...valid, extra: 42 };
    expect(validateTensorJson(extra) as unknown as Record<string, unknown>).toEqual(extra);
  });

  it('accepts empty shape and stride (scalar tensor)', () => {
    expect(validateTensorJson({ ...valid, shape: [], stride: [] }).shape).toEqual([]);
  });

  it('accepts a zero-byte tensor', () => {
    expect(validateTensorJson({ ...valid, size_bytes: 0, blob: '' }).size_bytes).toBe(0);
  });

  const nonObjects: [string, unknown][] = [
    ['null', null],
    ['undefined', undefined],
    ['a string', '{"dtype":"float32"}'],
    ['a number', 3],
    ['a boolean', true],
    ['a function', () => valid],
  ];
  for (const [label, value] of nonObjects) {
    it(`rejects ${label}`, () => {
      expect(() => validateTensorJson(value)).toThrow(P10Error);
      expect(() => validateTensorJson(value)).toThrow(/is not an object/);
    });
  }

  const badFields: [string, unknown, RegExp][] = [
    ['dtype missing', { ...valid, dtype: undefined }, /'dtype'/],
    ['dtype not a string', { ...valid, dtype: 7 }, /'dtype'/],
    ['shape missing', { ...valid, shape: undefined }, /'shape'/],
    ['shape not an array', { ...valid, shape: '2,3' }, /'shape'/],
    ['stride missing', { ...valid, stride: undefined }, /'stride'/],
    ['stride not an array', { ...valid, stride: 3 }, /'stride'/],
    ['size_bytes missing', { ...valid, size_bytes: undefined }, /'size_bytes'/],
    ['size_bytes not a number', { ...valid, size_bytes: '24' }, /'size_bytes'/],
    ['size_bytes fractional', { ...valid, size_bytes: 1.5 }, /'size_bytes'/],
    ['size_bytes negative', { ...valid, size_bytes: -1 }, /'size_bytes'/],
    ['encoding missing', { ...valid, encoding: undefined }, /'encoding'/],
    ['encoding unknown', { ...valid, encoding: 'base64+lz4' }, /'encoding'/],
    ['blob missing', { ...valid, blob: undefined }, /'blob'/],
    ['blob not a string', { ...valid, blob: [1, 2, 3] }, /'blob'/],
  ];
  for (const [label, value, message] of badFields) {
    it(`rejects when ${label}`, () => {
      expect(() => validateTensorJson(value)).toThrow(P10Error);
      expect(() => validateTensorJson(value)).toThrow(message);
    });
  }

  it('reports the first invalid field when several are wrong', () => {
    expect(() => validateTensorJson({})).toThrow(/'dtype'/);
  });

  it('rejects an array (not a tensor object)', () => {
    expect(() => validateTensorJson([1, 2, 3])).toThrow(P10Error);
  });

  it('validates the output of parseTensorJson', () => {
    const src = new Float32Array([1, 2, 3]);
    const parsed = parseTensorJson(makeJson('float32', [3], src));
    expect(validateTensorJson(parsed)).toBe(parsed);
  });
});

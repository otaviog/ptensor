import { describe, it, expect } from 'bun:test';
import { dtypeToNumber, numberToDtype } from '../dtype.js';
import { getDtypeFromTypedArray, createTypedArray } from '../typedArray.js';

// ------------------------------------------------------------------ //
// dtype.ts
// ------------------------------------------------------------------ //

describe('dtypeToNumber', () => {
  it('covers all expected dtypes', () => {
    const expected: Record<string, number> = {
      float32: 0, float64: 1, float16: 2,
      uint8: 3, uint16: 4, uint32: 5,
      int8: 6, int16: 7, int32: 8, int64: 9,
    };
    expect(dtypeToNumber).toEqual(expected);
  });
});

describe('numberToDtype', () => {
  it('is the inverse of dtypeToNumber', () => {
    for (const [name, code] of Object.entries(dtypeToNumber)) {
      expect(numberToDtype[code]).toBe(name);
    }
  });
});

// ------------------------------------------------------------------ //
// typedArray.ts
// ------------------------------------------------------------------ //

describe('getDtypeFromTypedArray', () => {
  const cases: [TypedArray, string][] = [
    [new Float32Array(0), 'float32'],
    [new Float64Array(0), 'float64'],
    [new Uint8Array(0),   'uint8'],
    [new Uint16Array(0),  'uint16'],
    [new Uint32Array(0),  'uint32'],
    [new Int8Array(0),    'int8'],
    [new Int16Array(0),   'int16'],
    [new Int32Array(0),   'int32'],
    [new BigInt64Array(0),'int64'],
  ];
  type TypedArray = (typeof cases)[0][0];

  for (const [arr, dtype] of cases) {
    it(`${arr.constructor.name} → '${dtype}'`, () => {
      expect(getDtypeFromTypedArray(arr)).toBe(dtype);
    });
  }
});

describe('createTypedArray', () => {
  const cases = [
    ['float32', Float32Array,  4],
    ['float64', Float64Array,  8],
    ['float16', Uint16Array,   2], // stored as bits
    ['uint8',   Uint8Array,    1],
    ['uint16',  Uint16Array,   2],
    ['uint32',  Uint32Array,   4],
    ['int8',    Int8Array,     1],
    ['int16',   Int16Array,    2],
    ['int32',   Int32Array,    4],
    ['int64',   BigInt64Array, 8],
  ] as const;

  for (const [dtype, Ctor, bytes] of cases) {
    it(`'${dtype}' creates ${Ctor.name} with correct byte size`, () => {
      const arr = createTypedArray(dtype, 4);
      expect(arr).toBeInstanceOf(Ctor);
      expect(arr.length).toBe(4);
      expect(arr.byteLength).toBe(4 * bytes);
    });
  }

  it('throws for an unknown dtype string', () => {
    expect(() => createTypedArray('bogus' as never, 1)).toThrow();
  });
});


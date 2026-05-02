import * as ffi from './backends/bun/ffi'
import { P10Error } from './p10Error';

const MAX_DIMS = 8;

export type Dtype = 'float32' | 'float64' | 'uint8' | 'uint16' | 'uint32' | 'int8' | 'int16' | 'int32' | 'int64';

export interface Tensor {
  getSize(): bigint;
  getShape(): bigint[];
  getStride(): bigint[];
  getDtype(): Dtype;
  delete(): void;
}

export type ArrayTypes = Float32Array | Float64Array | Uint8Array | Uint16Array | Int16Array | Int32Array | Uint32Array | BigInt64Array;

const arrayToDtype: Record<string, Dtype> = {
  Float32Array: 'float32',
  Float64Array: 'float64',
  Uint8Array: 'uint8',
  Uint16Array: 'uint16',
  Int16Array: 'int16',
  Int32Array: 'int32',
  Uint32Array: 'uint32',
  BigInt64Array: 'int64'

}

const dtypeToNumber: Record<Dtype, number> = {
  'float32': 0,
  'float64': 1,
  'uint8': 3,
  'uint16': 4,
  'uint32': 5,
  'int8': 6,
  'int16': 7,
  'int32': 8,
  'int64': 9
};


const numberToDtype: Record<number, Dtype> = Object.fromEntries(
  Object.entries(dtypeToNumber).map(([key, value]) => [value, key]));

export function fromData(data: ArrayTypes, shape: number[]) {
  let tensor: number = 0;

  ffi.p10_from_data(
    tensor,
    dtypeToNumber[arrayToDtype[typeof data]],
    shape,
    shape.length,
    data
  )

  return {
    getShape: (): bigint[] => {
      const array = new BigInt64Array(MAX_DIMS);

      ffi.p10_get_shape(tensor, array, array.length);
      return Array.from(array);
    },
    getDtype: (): Dtype {
      const dtypeNum = ffi.p10_get_dtype();
      if (!(dtypeNum in numberToDtype)) {
        throw new P10Error(`Invalid result from ffi ${dtypeNum}`)
      }
      return numberToDtype[dtypeNum];
    },
    
    
  } as Tensor;
}



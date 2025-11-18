import { DTypeString } from './dtype.js';

export type TypedArrayType =
    | Float32Array
    | Float64Array
    | Uint8Array
    | Uint16Array
    | Uint32Array
    | Int8Array
    | Int16Array
    | Int32Array
    | BigInt64Array;

export const getDtypeFromTypedArray = (data: TypedArrayType): DTypeString => {
    if (data instanceof Float32Array) return 'float32';
    if (data instanceof Float64Array) return 'float64';
    if (data instanceof Uint8Array) return 'uint8';
    if (data instanceof Uint16Array) return 'uint16';
    if (data instanceof Uint32Array) return 'uint32';
    if (data instanceof Int8Array) return 'int8';
    if (data instanceof Int16Array) return 'int16';
    if (data instanceof Int32Array) return 'int32';
    if (data instanceof BigInt64Array) return 'int64';
    throw new Error('Unsupported typed array type');
}

export const createTypedArray = (dtype: DTypeString, size: number): TypedArrayType => {
    switch (dtype) {
        case 'float32': return new Float32Array(size);
        case 'float64': return new Float64Array(size);
        case 'uint8': return new Uint8Array(size);
        case 'uint16': return new Uint16Array(size);
        case 'uint32': return new Uint32Array(size);
        case 'int8': return new Int8Array(size);
        case 'int16': return new Int16Array(size);
        case 'int32': return new Int32Array(size);
        case 'int64': return new BigInt64Array(size);
        default: throw new Error(`Unsupported dtype: ${dtype}`);
    }
}

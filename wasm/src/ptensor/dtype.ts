import { MODULE } from './module-init.js';

export type DTypeString = 'float32' | 'float64' | 'uint8' | 'uint16' | 'uint32' | 'int8' | 'int16' | 'int32' | 'int64';

export const parseDtype = (dtypeStr: string): DTypeString => {
    const lower = dtypeStr.toLowerCase();
    if (lower.includes('float32')) return 'float32';
    if (lower.includes('float64')) return 'float64';
    if (lower.includes('uint8')) return 'uint8';
    if (lower.includes('uint16')) return 'uint16';
    if (lower.includes('uint32')) return 'uint32';
    if (lower.includes('int8')) return 'int8';
    if (lower.includes('int16')) return 'int16';
    if (lower.includes('int32')) return 'int32';
    if (lower.includes('int64')) return 'int64';
    return 'float32'; // default
}

export const createWasmDtype = (dtype: DTypeString): any => {
    const dtypeCode = getDtypeCode(dtype);
    return new (MODULE.Dtype as any)(dtypeCode);
}

const getDtypeCode = (dtype: DTypeString): number => {
    // Map string dtype to DtypeCode enum values (from ptensor_dtype.h)
    const dtypeMap: Record<DTypeString, number> = {
        'float32': 0,  // P10_DTYPE_FLOAT32
        'float64': 1,  // P10_DTYPE_FLOAT64
        'uint8': 3,    // P10_DTYPE_UINT8
        'uint16': 4,   // P10_DTYPE_UINT16
        'uint32': 5,   // P10_DTYPE_UINT32
        'int8': 6,     // P10_DTYPE_INT8
        'int16': 7,    // P10_DTYPE_INT16
        'int32': 8,    // P10_DTYPE_INT32
        'int64': 9     // P10_DTYPE_INT64
    };
    return dtypeMap[dtype];
}

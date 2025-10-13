/**
 * Data types supported by Ptensor
 */
export enum DType {
  FLOAT32 = 0,
  FLOAT64 = 1,
  FLOAT16 = 2,
  UINT8 = 3,
  UINT16 = 4,
  UINT32 = 5,
  INT8 = 6,
  INT16 = 7,
  INT32 = 8,
  INT64 = 9,
}

/**
 * Error codes returned by C API
 */
export enum ErrorCode {
  OK = 0,
  UNKNOWN_ERROR = 1,
  ASSERTION_ERROR = 2,
  INVALID_ARGUMENT = 3,
  INVALID_OPERATION = 4,
  OUT_OF_MEMORY = 5,
  OUT_OF_RANGE = 6,
  NOT_IMPLEMENTED = 7,
  OS_ERROR = 8,
  IO_ERROR = 9,
}

/**
 * Device types
 */
export enum Device {
  CPU = 0,
  CUDA = 1,
  OCL = 2,
}

/**
 * Map error codes to error messages
 */
export const ERROR_MESSAGES: Record<ErrorCode, string> = {
  [ErrorCode.OK]: 'Success',
  [ErrorCode.UNKNOWN_ERROR]: 'Unknown error',
  [ErrorCode.ASSERTION_ERROR]: 'Assertion failed',
  [ErrorCode.INVALID_ARGUMENT]: 'Invalid argument',
  [ErrorCode.INVALID_OPERATION]: 'Invalid operation',
  [ErrorCode.OUT_OF_MEMORY]: 'Out of memory',
  [ErrorCode.OUT_OF_RANGE]: 'Out of range',
  [ErrorCode.NOT_IMPLEMENTED]: 'Not implemented',
  [ErrorCode.OS_ERROR]: 'Operating system error',
  [ErrorCode.IO_ERROR]: 'Input/output error',
};

/**
 * Map DType to TypedArray constructors
 */

export const dtype_to_typed_array = (dtype: DType) => {
  if (dtype === DType.FLOAT32) return Float32Array;
  if (dtype === DType.FLOAT64) return Float64Array;
  if (dtype === DType.UINT8) return Uint8Array;
  if (dtype === DType.UINT16) return Uint16Array;
  if (dtype === DType.UINT32) return Uint32Array;
  if (dtype === DType.INT8) return Int8Array;
  if (dtype === DType.INT16) return Int16Array;
  if (dtype === DType.INT32) return Int32Array;

  throw new Error(`Invalid dtype ${dtype}`)
}

/**
 * Get byte size for a given dtype
 */
export function getDTypeSize(dtype: DType): number {
  switch (dtype) {
    case DType.FLOAT32:
    case DType.UINT32:
    case DType.INT32:
      return 4;
    case DType.FLOAT64:
    case DType.INT64:
      return 8;
    case DType.FLOAT16:
    case DType.UINT16:
    case DType.INT16:
      return 2;
    case DType.UINT8:
    case DType.INT8:
      return 1;
    default:
      throw new Error(`Unknown dtype: ${dtype}`);
  }
}

/**
 * Internal FFI utilities shared between tensor.ts and infer.ts.
 *
 * Opaque C handles (Ptensor, P10Infer, …) are stored as BigUint64Array(1)
 * so that the C library can write pointer values into them (via Ptensor*
 * out-params) and so we can pass them back as raw-integer input args.
 */

/** Coerces an unknown FFI return value to a number (int / enum). */
export function ffiInt(v: unknown): number {
  return v as number;
}

/** Coerces an unknown FFI return value to a bigint (size_t / u64). */
export function ffiU64(v: unknown): bigint {
  const n = v as bigint | number;
  return typeof n === 'bigint' ? n : BigInt(n);
}

/**
 * Reads an opaque void* handle out of a BigUint64Array slot and returns
 * it as a JS number suitable for passing as a raw-pointer FFI input arg.
 *
 * Bun FFI accepts `number` for `FFIType.ptr` input params and interprets
 * it as a raw 64-bit pointer value.
 */
export function readHandle(buf: BigUint64Array): number {
  return Number(buf[0]);
}

/** Allocates a zeroed BigUint64Array(1) to receive a C out-pointer. */
export function newHandleBuf(): BigUint64Array {
  return new BigUint64Array(1);
}

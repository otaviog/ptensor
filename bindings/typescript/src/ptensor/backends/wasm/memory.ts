import type { PTensorWasmModule } from './module';

// ------------------------------------------------------------------ //
// Stack-based out-param helpers
// ------------------------------------------------------------------ //

/**
 * Allocates a 4-byte (i32) slot on the Emscripten stack, invokes `cb` with
 * the slot address, then returns both the callback's return value and the
 * i32 written into the slot by the callee.
 *
 * Use this for C out-params of type `void**` / `Ptensor*` / `P10Infer*`.
 */
export function withPtrOutSlot<T>(
  mod: PTensorWasmModule,
  cb: (slot: number) => T,
): [cbResult: T, slotValue: number] {
  const sp = mod.stackSave();
  const slot = mod.stackAlloc(4);
  mod.setValue(slot, 0, 'i32');
  const cbResult = cb(slot);
  const slotValue = mod.getValue(slot, 'i32');
  mod.stackRestore(sp);
  return [cbResult, slotValue];
}

// ------------------------------------------------------------------ //
// Heap allocation helpers
// ------------------------------------------------------------------ //

/**
 * Copies an `ArrayBufferView` into fresh WASM heap memory.
 * The caller is responsible for calling `mod._free(ptr)` when done.
 */
export function mallocBuffer(mod: PTensorWasmModule, data: ArrayBufferView): number {
  const ptr = mod._malloc(data.byteLength);
  if (!ptr) throw new Error('WASM _malloc failed (out of memory)');
  mod.HEAPU8.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), ptr);
  return ptr;
}

/**
 * Writes an array of bigint values as packed int64 on the WASM heap.
 * The caller is responsible for calling `mod._free(ptr)` when done.
 */
export function mallocI64Array(mod: PTensorWasmModule, values: bigint[]): number {
  const byteLen = values.length * 8;
  const ptr = mod._malloc(byteLen);
  if (!ptr) throw new Error('WASM _malloc failed (out of memory)');
  const view = new BigInt64Array(mod.HEAPU8.buffer, ptr, values.length);
  for (let i = 0; i < values.length; i++) view[i] = values[i];
  return ptr;
}

/**
 * Reads `n` int64 values from the WASM heap at `ptr`.
 * Does NOT free the pointer.
 */
export function readI64Array(mod: PTensorWasmModule, ptr: number, n: number): bigint[] {
  const view = new BigInt64Array(mod.HEAPU8.buffer, ptr, n);
  return Array.from(view);
}

// ------------------------------------------------------------------ //
// String helpers
// ------------------------------------------------------------------ //

/**
 * Reads the last C API error message. Returns `null` when the pointer is 0.
 * The returned pointer is owned by the C library and must not be freed.
 */
export function readLastErrorMessage(mod: PTensorWasmModule): string | null {
  const ptr = mod._p10_get_last_error_message();
  return ptr ? mod.UTF8ToString(ptr) : null;
}

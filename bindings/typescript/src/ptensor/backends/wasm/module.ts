/**
 * TypeScript type for the Emscripten-generated PTensorModule.
 *
 * Only the tensor-core subset of the C API is declared here;
 * media and infer are not available in the WASM build.
 */
export interface PTensorWasmModule {
  // ------------------------------------------------------------------ //
  // Emscripten runtime utilities
  // ------------------------------------------------------------------ //

  stackSave(): number;
  stackRestore(sp: number): void;
  /** Allocates `bytes` on the Emscripten stack (always 16-byte aligned). */
  stackAlloc(bytes: number): number;

  /** Reads a value from the WASM heap. `type` is an Emscripten type string ('i32', 'i64', etc.) */
  getValue(ptr: number, type: string): number;
  /** Writes a value to the WASM heap. */
  setValue(ptr: number, value: number, type: string): void;

  /** Reads a null-terminated UTF-8 string from the WASM heap. */
  UTF8ToString(ptr: number): string;

  /** Raw byte view of the entire WASM memory. */
  HEAPU8: Uint8Array;

  /** Allocates `size` bytes on the WASM heap. Returns 0 on failure. */
  _malloc(size: number): number;
  /** Frees a pointer previously returned by `_malloc`. */
  _free(ptr: number): void;

  // ------------------------------------------------------------------ //
  // Error
  // ------------------------------------------------------------------ //

  /** Returns a pointer to the last error message C-string, or 0 if none. */
  _p10_get_last_error_message(): number;

  // ------------------------------------------------------------------ //
  // Dtype
  // ------------------------------------------------------------------ //

  /** Returns a pointer to the dtype name C-string (e.g. "float32"). */
  _p10_dtype_to_string(dtype: number): number;
  /** Returns the size in bytes of one element of the given dtype. */
  _p10_dtype_size_bytes(dtype: number): number;

  // ------------------------------------------------------------------ //
  // Tensor lifecycle
  // ------------------------------------------------------------------ //

  /** p10_from_data — writes the new Ptensor handle to *out. Returns P10ErrorEnum. */
  _p10_from_data(
    out: number,       // Ptensor* (pointer slot on WASM heap)
    dtype: number,     // P10DTypeEnum
    shape: number,     // const int64_t* (WASM heap)
    numDims: number,   // size_t
    data: number       // void*  (WASM heap)
  ): number;

  /** p10_from_data_strided. Returns P10ErrorEnum. */
  _p10_from_data_strided(
    out: number,
    dtype: number,
    shape: number,
    strides: number,   // const int64_t* (WASM heap)
    numDims: number,
    data: number
  ): number;

  /** p10_destroy — sets *tensorPtr to NULL and frees the tensor. Returns P10ErrorEnum. */
  _p10_destroy(tensorPtr: number): number;

  // ------------------------------------------------------------------ //
  // Tensor accessors
  // ------------------------------------------------------------------ //

  _p10_get_size(handle: number): number;
  _p10_get_size_bytes(handle: number): number;
  _p10_get_dtype(handle: number): number;
  /** p10_get_shape — fills shapePtr with ndim int64 values. Returns P10ErrorEnum. */
  _p10_get_shape(handle: number, shapePtr: number, numDims: number): number;
  /** p10_get_stride — fills stridesPtr with ndim int64 values. Returns P10ErrorEnum. */
  _p10_get_stride(handle: number, stridesPtr: number, numDims: number): number;
  _p10_get_ndim(handle: number): number;
  _p10_get_data(handle: number): number;
  _p10_is_empty(handle: number): number;
}

/**
 * The Emscripten-generated factory function.
 * Import from the generated `ptensor.js` loader and call to instantiate the module.
 */
export type PTensorModuleFactory = (
  opts?: Record<string, unknown>
) => Promise<PTensorWasmModule>;

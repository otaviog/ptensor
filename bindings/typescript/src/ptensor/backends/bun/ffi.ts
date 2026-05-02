import { dlopen, FFIType, suffix } from "bun:ffi";

const libPath = process.env["PTENSOR_LIB_PATH"] ?? `libptensor_capi.${suffix}`;

export const {
  symbols: {
    // error
    p10_get_last_error_message,
    // dtype
    p10_dtype_to_string,
    p10_dtype_from_string,
    p10_dtype_size_bytes,
    // tensor lifecycle
    p10_from_data,
    p10_from_data_strided,
    p10_destroy,
    // tensor accessors
    p10_get_size,
    p10_get_size_bytes,
    p10_get_dtype,
    p10_get_shape,
    p10_get_stride,
    p10_get_ndim,
    p10_get_data,
    p10_is_empty,
  },
} = dlopen(libPath, {
  // ------------------------------------------------------------------ //
  // Error
  // ------------------------------------------------------------------ //

  /** Returns the last error message string, or null if none. */
  p10_get_last_error_message: {
    args: [],
    returns: FFIType.cstring,
  },

  // ------------------------------------------------------------------ //
  // DType
  // ------------------------------------------------------------------ //

  /** Converts a dtype enum value to its canonical string (e.g. "float32"). */
  p10_dtype_to_string: {
    args: [FFIType.int],   // P10DTypeEnum
    returns: FFIType.cstring,
  },

  /** Parses a dtype string into a dtype enum value. Returns a P10ErrorEnum. */
  p10_dtype_from_string: {
    args: [
      FFIType.cstring,   // const char* type_str
      FFIType.ptr,       // P10DTypeEnum* out_dtype
    ],
    returns: FFIType.int, // P10ErrorEnum
  },

  /** Returns the byte size of a single element for the given dtype. */
  p10_dtype_size_bytes: {
    args: [FFIType.int],  // P10DTypeEnum
    returns: FFIType.u64, // size_t
  },

  // ------------------------------------------------------------------ //
  // Tensor – lifecycle
  // ------------------------------------------------------------------ //

  /**
   * Creates a tensor view over an existing contiguous data buffer.
   * The caller must keep the buffer alive while the tensor is in use.
   * Returns a P10ErrorEnum.
   */
  p10_from_data: {
    args: [
      FFIType.ptr,  // Ptensor* out
      FFIType.int,  // P10DTypeEnum dtype
      FFIType.ptr,  // const int64_t* shape
      FFIType.u64,  // size_t num_dims
      FFIType.ptr,  // void* data
    ],
    returns: FFIType.int, // P10ErrorEnum
  },

  /**
   * Creates a tensor view with custom per-element strides.
   * Strides are in element counts, not bytes.
   * Returns a P10ErrorEnum.
   */
  p10_from_data_strided: {
    args: [
      FFIType.ptr,  // Ptensor* out
      FFIType.int,  // P10DTypeEnum dtype
      FFIType.ptr,  // const int64_t* shape
      FFIType.ptr,  // const int64_t* strides
      FFIType.u64,  // size_t num_dims
      FFIType.ptr,  // void* data
    ],
    returns: FFIType.int, // P10ErrorEnum
  },

  /**
   * Destroys a tensor and sets *tensor to NULL.
   * Returns a P10ErrorEnum (always OK).
   */
  p10_destroy: {
    args: [FFIType.ptr],  // Ptensor*
    returns: FFIType.int, // P10ErrorEnum
  },

  // ------------------------------------------------------------------ //
  // Tensor – accessors
  // ------------------------------------------------------------------ //

  /** Returns the total number of elements. */
  p10_get_size: {
    args: [FFIType.ptr],  // Ptensor
    returns: FFIType.u64, // size_t
  },

  /** Returns the total size of the tensor data in bytes. */
  p10_get_size_bytes: {
    args: [FFIType.ptr],  // Ptensor
    returns: FFIType.u64, // size_t
  },

  /** Returns the data type enum value of the tensor. */
  p10_get_dtype: {
    args: [FFIType.ptr],  // Ptensor
    returns: FFIType.int, // P10DTypeEnum
  },

  /**
   * Fills *shape with up to num_dims dimension sizes (int64_t each).
   * Returns a P10ErrorEnum.
   */
  p10_get_shape: {
    args: [
      FFIType.ptr,  // Ptensor
      FFIType.ptr,  // int64_t* shape
      FFIType.u64,  // size_t num_dims
    ],
    returns: FFIType.int, // P10ErrorEnum
  },

  /**
   * Fills *strides with up to num_dims per-element strides (int64_t each).
   * Returns a P10ErrorEnum.
   */
  p10_get_stride: {
    args: [
      FFIType.ptr,  // Ptensor
      FFIType.ptr,  // int64_t* strides
      FFIType.u64,  // size_t num_dims
    ],
    returns: FFIType.int, // P10ErrorEnum
  },

  /** Returns the number of dimensions (rank) of the tensor. */
  p10_get_ndim: {
    args: [FFIType.ptr],  // Ptensor
    returns: FFIType.u64, // size_t
  },

  /** Returns a raw pointer to the tensor's data buffer. */
  p10_get_data: {
    args: [FFIType.ptr],  // Ptensor
    returns: FFIType.ptr, // void*
  },

  /** Returns 1 if the tensor has no elements, 0 otherwise. */
  p10_is_empty: {
    args: [FFIType.ptr],  // Ptensor
    returns: FFIType.int,
  },
});


import { dlopen, FFIType, suffix } from 'bun:ffi';

const libPath = process.env.PTENSOR_LIB_PATH ?? `libptensor_capi.${suffix}`;

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
    // infer
    p10_infer_from_onnx,
    p10_infer_destroy,
    p10_infer_get_input_count,
    p10_infer_get_output_count,
    p10_infer_run,
    // media — VideoFrame
    p10_video_frame_create,
    p10_video_frame_destroy,
    p10_video_frame_width,
    p10_video_frame_height,
    p10_video_frame_channels,
    p10_video_frame_image,
    p10_video_frame_time_base_num,
    p10_video_frame_time_base_den,
    p10_video_frame_time_stamp,
    p10_video_frame_set_time_parts,
    // media — AudioFrame
    p10_audio_frame_create,
    p10_audio_frame_destroy,
    p10_audio_frame_samples_count,
    p10_audio_frame_channels_count,
    p10_audio_frame_sample_rate,
    p10_audio_frame_duration_seconds,
    p10_audio_frame_samples,
    p10_audio_frame_time_base_num,
    p10_audio_frame_time_base_den,
    p10_audio_frame_time_stamp,
    p10_audio_frame_set_time_parts,
    // media — MediaCapture
    p10_media_capture_open,
    p10_media_capture_close,
    p10_media_capture_next_frame,
    p10_media_capture_get_video,
    p10_media_capture_get_audio,
    p10_media_capture_video_width,
    p10_media_capture_video_height,
    p10_media_capture_video_frame_rate_num,
    p10_media_capture_video_frame_rate_den,
    p10_media_capture_audio_sample_rate,
    p10_media_capture_audio_channels,
    p10_media_capture_video_frame_count,
    p10_media_capture_duration,
    // media — MediaWriter
    p10_media_writer_open_ffi,
    p10_media_writer_close,
    p10_media_writer_write_video,
    p10_media_writer_write_audio,
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
    args: [FFIType.int], // P10DTypeEnum
    returns: FFIType.cstring,
  },

  /** Parses a dtype string into a dtype enum value. Returns a P10ErrorEnum. */
  p10_dtype_from_string: {
    args: [
      FFIType.cstring, // const char* type_str
      FFIType.ptr, // P10DTypeEnum* out_dtype
    ],
    returns: FFIType.int, // P10ErrorEnum
  },

  /** Returns the byte size of a single element for the given dtype. */
  p10_dtype_size_bytes: {
    args: [FFIType.int], // P10DTypeEnum
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
      FFIType.ptr, // Ptensor* out
      FFIType.int, // P10DTypeEnum dtype
      FFIType.ptr, // const int64_t* shape
      FFIType.u64, // size_t num_dims
      FFIType.ptr, // void* data
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
      FFIType.ptr, // Ptensor* out
      FFIType.int, // P10DTypeEnum dtype
      FFIType.ptr, // const int64_t* shape
      FFIType.ptr, // const int64_t* strides
      FFIType.u64, // size_t num_dims
      FFIType.ptr, // void* data
    ],
    returns: FFIType.int, // P10ErrorEnum
  },

  /**
   * Destroys a tensor and sets *tensor to NULL.
   * Returns a P10ErrorEnum (always OK).
   */
  p10_destroy: {
    args: [FFIType.ptr], // Ptensor*
    returns: FFIType.int, // P10ErrorEnum
  },

  // ------------------------------------------------------------------ //
  // Tensor – accessors
  // ------------------------------------------------------------------ //

  /** Returns the total number of elements. */
  p10_get_size: {
    args: [FFIType.ptr], // Ptensor
    returns: FFIType.u64, // size_t
  },

  /** Returns the total size of the tensor data in bytes. */
  p10_get_size_bytes: {
    args: [FFIType.ptr], // Ptensor
    returns: FFIType.u64, // size_t
  },

  /** Returns the data type enum value of the tensor. */
  p10_get_dtype: {
    args: [FFIType.ptr], // Ptensor
    returns: FFIType.int, // P10DTypeEnum
  },

  /**
   * Fills *shape with up to num_dims dimension sizes (int64_t each).
   * Returns a P10ErrorEnum.
   */
  p10_get_shape: {
    args: [
      FFIType.ptr, // Ptensor
      FFIType.ptr, // int64_t* shape
      FFIType.u64, // size_t num_dims
    ],
    returns: FFIType.int, // P10ErrorEnum
  },

  /**
   * Fills *strides with up to num_dims per-element strides (int64_t each).
   * Returns a P10ErrorEnum.
   */
  p10_get_stride: {
    args: [
      FFIType.ptr, // Ptensor
      FFIType.ptr, // int64_t* strides
      FFIType.u64, // size_t num_dims
    ],
    returns: FFIType.int, // P10ErrorEnum
  },

  /** Returns the number of dimensions (rank) of the tensor. */
  p10_get_ndim: {
    args: [FFIType.ptr], // Ptensor
    returns: FFIType.u64, // size_t
  },

  /** Returns a raw pointer to the tensor's data buffer. */
  p10_get_data: {
    args: [FFIType.ptr], // Ptensor
    returns: FFIType.ptr, // void*
  },

  /** Returns 1 if the tensor has no elements, 0 otherwise. */
  p10_is_empty: {
    args: [FFIType.ptr], // Ptensor
    returns: FFIType.int,
  },

  // ------------------------------------------------------------------ //
  // Infer
  // ------------------------------------------------------------------ //

  /** Creates an inference session from an ONNX model file path. Returns P10ErrorEnum. */
  p10_infer_from_onnx: {
    args: [
      FFIType.ptr, // P10Infer* out
      FFIType.cstring, // const char* onnx_model_path
    ],
    returns: FFIType.int,
  },

  /** Destroys an inference session. Sets *infer to NULL. Returns P10ErrorEnum. */
  p10_infer_destroy: {
    args: [FFIType.ptr], // P10Infer*
    returns: FFIType.int,
  },

  /** Returns the number of input tensors expected by the model. */
  p10_infer_get_input_count: {
    args: [FFIType.ptr], // P10Infer (handle value)
    returns: FFIType.u64,
  },

  /** Returns the number of output tensors produced by the model. */
  p10_infer_get_output_count: {
    args: [FFIType.ptr], // P10Infer (handle value)
    returns: FFIType.u64,
  },

  /**
   * Runs inference.
   *   infer          - session handle value
   *   input_tensors  - ptr to contiguous array of Ptensor handle values
   *   num_inputs     - must match p10_infer_get_input_count()
   *   output_tensors - ptr to contiguous array of Ptensor slots (filled on success)
   *   num_outputs    - must match p10_infer_get_output_count()
   */
  p10_infer_run: {
    args: [
      FFIType.ptr, // P10Infer infer
      FFIType.ptr, // const Ptensor* input_tensors
      FFIType.u64, // size_t num_inputs
      FFIType.ptr, // Ptensor* output_tensors
      FFIType.u64, // size_t num_outputs
    ],
    returns: FFIType.int,
  },

  // ------------------------------------------------------------------ //
  // Media — VideoFrame
  // ------------------------------------------------------------------ //

  p10_video_frame_create: {
    args: [FFIType.ptr, FFIType.u64, FFIType.u64], // P10VideoFrame*, width, height
    returns: FFIType.int,
  },
  p10_video_frame_destroy: {
    args: [FFIType.ptr], // P10VideoFrame*
    returns: FFIType.int,
  },
  p10_video_frame_width: {
    args: [FFIType.ptr], // P10VideoFrame
    returns: FFIType.u64,
  },
  p10_video_frame_height: {
    args: [FFIType.ptr], // P10VideoFrame
    returns: FFIType.u64,
  },
  p10_video_frame_channels: {
    args: [FFIType.ptr], // P10VideoFrame
    returns: FFIType.u64,
  },
  p10_video_frame_image: {
    args: [FFIType.ptr, FFIType.ptr], // P10VideoFrame, Ptensor*
    returns: FFIType.int,
  },
  p10_video_frame_time_base_num: {
    args: [FFIType.ptr], // P10VideoFrame
    returns: FFIType.i64,
  },
  p10_video_frame_time_base_den: {
    args: [FFIType.ptr],
    returns: FFIType.i64,
  },
  p10_video_frame_time_stamp: {
    args: [FFIType.ptr],
    returns: FFIType.i64,
  },
  p10_video_frame_set_time_parts: {
    args: [FFIType.ptr, FFIType.i64, FFIType.i64, FFIType.i64],
    returns: FFIType.void,
  },

  // ------------------------------------------------------------------ //
  // Media — AudioFrame
  // ------------------------------------------------------------------ //

  p10_audio_frame_create: {
    args: [FFIType.ptr, FFIType.ptr, FFIType.u64], // P10AudioFrame*, Ptensor, sample_rate
    returns: FFIType.int,
  },
  p10_audio_frame_destroy: {
    args: [FFIType.ptr], // P10AudioFrame*
    returns: FFIType.int,
  },
  p10_audio_frame_samples_count: {
    args: [FFIType.ptr],
    returns: FFIType.i64,
  },
  p10_audio_frame_channels_count: {
    args: [FFIType.ptr],
    returns: FFIType.i64,
  },
  p10_audio_frame_sample_rate: {
    args: [FFIType.ptr],
    returns: FFIType.u64,
  },
  p10_audio_frame_duration_seconds: {
    args: [FFIType.ptr],
    returns: FFIType.f64,
  },
  p10_audio_frame_samples: {
    args: [FFIType.ptr, FFIType.ptr], // P10AudioFrame, Ptensor*
    returns: FFIType.int,
  },
  p10_audio_frame_time_base_num: {
    args: [FFIType.ptr],
    returns: FFIType.i64,
  },
  p10_audio_frame_time_base_den: {
    args: [FFIType.ptr],
    returns: FFIType.i64,
  },
  p10_audio_frame_time_stamp: {
    args: [FFIType.ptr],
    returns: FFIType.i64,
  },
  p10_audio_frame_set_time_parts: {
    args: [FFIType.ptr, FFIType.i64, FFIType.i64, FFIType.i64],
    returns: FFIType.void,
  },

  // ------------------------------------------------------------------ //
  // Media — MediaCapture
  // ------------------------------------------------------------------ //

  p10_media_capture_open: {
    args: [FFIType.ptr, FFIType.cstring], // P10MediaCapture*, path
    returns: FFIType.int,
  },
  p10_media_capture_close: {
    args: [FFIType.ptr], // P10MediaCapture*
    returns: FFIType.int,
  },
  p10_media_capture_next_frame: {
    args: [FFIType.ptr, FFIType.ptr], // P10MediaCapture, int*
    returns: FFIType.int,
  },
  p10_media_capture_get_video: {
    args: [FFIType.ptr, FFIType.ptr], // P10MediaCapture, P10VideoFrame*
    returns: FFIType.int,
  },
  p10_media_capture_get_audio: {
    args: [FFIType.ptr, FFIType.ptr], // P10MediaCapture, P10AudioFrame*
    returns: FFIType.int,
  },
  p10_media_capture_video_width: {
    args: [FFIType.ptr],
    returns: FFIType.i32,
  },
  p10_media_capture_video_height: {
    args: [FFIType.ptr],
    returns: FFIType.i32,
  },
  p10_media_capture_video_frame_rate_num: {
    args: [FFIType.ptr],
    returns: FFIType.i64,
  },
  p10_media_capture_video_frame_rate_den: {
    args: [FFIType.ptr],
    returns: FFIType.i64,
  },
  p10_media_capture_audio_sample_rate: {
    args: [FFIType.ptr],
    returns: FFIType.f64,
  },
  p10_media_capture_audio_channels: {
    args: [FFIType.ptr],
    returns: FFIType.u64,
  },
  p10_media_capture_video_frame_count: {
    args: [FFIType.ptr],
    returns: FFIType.i64,
  },
  p10_media_capture_duration: {
    args: [FFIType.ptr],
    returns: FFIType.f64,
  },

  // ------------------------------------------------------------------ //
  // Media — MediaWriter
  // ------------------------------------------------------------------ //

  /** FFI variant: frame_rate passed as separate num/den instead of P10Rational. */
  p10_media_writer_open_ffi: {
    args: [
      FFIType.ptr, // P10MediaWriter*
      FFIType.cstring, // const char* path
      FFIType.i32, // width
      FFIType.i32, // height
      FFIType.i64, // frame_rate_num
      FFIType.i64, // frame_rate_den
      FFIType.f64, // audio_sample_rate_hz
      FFIType.u64, // audio_channels
    ],
    returns: FFIType.int,
  },
  p10_media_writer_close: {
    args: [FFIType.ptr], // P10MediaWriter*
    returns: FFIType.int,
  },
  p10_media_writer_write_video: {
    args: [FFIType.ptr, FFIType.ptr], // P10MediaWriter, P10VideoFrame
    returns: FFIType.int,
  },
  p10_media_writer_write_audio: {
    args: [FFIType.ptr, FFIType.ptr], // P10MediaWriter, P10AudioFrame
    returns: FFIType.int,
  },
});

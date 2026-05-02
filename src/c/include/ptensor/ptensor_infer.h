#ifndef PTENSOR_INFER_H_
#define PTENSOR_INFER_H_

#include <stddef.h>

#include "config.h"
#include "ptensor_error.h"
#include "ptensor_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle for an inference session.
typedef void* P10Infer;

/// Creates an inference session from an ONNX model file path.
///
/// The engine defaults to ONNX Runtime. The model path may be a local
/// file path or a URL (downloaded and cached automatically).
///
/// On success, writes the handle to *infer. The caller must call
/// p10_infer_destroy() to release resources.
PTENSOR_API P10ErrorEnum p10_infer_from_onnx(P10Infer* infer, const char* onnx_model_path);

/// Destroys an inference session and releases all resources.
/// Sets *infer to NULL.
PTENSOR_API P10ErrorEnum p10_infer_destroy(P10Infer* infer);

/// Returns the number of input tensors expected by the model.
PTENSOR_API size_t p10_infer_get_input_count(P10Infer infer);

/// Returns the number of output tensors produced by the model.
PTENSOR_API size_t p10_infer_get_output_count(P10Infer infer);

/// Runs inference.
///
/// input_tensors  - array of num_inputs Ptensor handles with the model inputs.
/// num_inputs     - must match p10_infer_get_input_count().
/// output_tensors - caller-provided array of num_outputs Ptensor slots.
///                  Each slot is set to a newly allocated Ptensor on success.
///                  The caller is responsible for calling p10_destroy() on
///                  each output handle when done.
/// num_outputs    - must match p10_infer_get_output_count().
PTENSOR_API P10ErrorEnum p10_infer_run(
    P10Infer infer,
    const Ptensor* input_tensors,
    size_t num_inputs,
    Ptensor* output_tensors,
    size_t num_outputs
);

#ifdef __cplusplus
}
#endif

#endif  // PTENSOR_INFER_H_

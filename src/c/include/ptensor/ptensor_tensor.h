#ifndef PTENSOR_TENSOR_H_
#define PTENSOR_TENSOR_H_

#include <stdint.h>
#include <stddef.h>

#include "config.h"
#include "ptensor_dtype.h"
#include "ptensor_error.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void* Ptensor;

/// Creates a tensor that is a view over an existing data buffer (contiguous layout).
///
/// The caller owns the data buffer and must ensure it remains valid for the lifetime
/// of the tensor.
PTENSOR_API P10ErrorEnum p10_from_data(
    Ptensor* tensor,
    P10DTypeEnum dtype,
    const int64_t* shape,
    size_t num_dims,
    void* data
);

/// Creates a tensor view with a custom per-element stride per dimension.
///
/// Strides are in element counts (not bytes). The caller owns the data buffer.
PTENSOR_API P10ErrorEnum p10_from_data_strided(
    Ptensor* tensor,
    P10DTypeEnum dtype,
    const int64_t* shape,
    const int64_t* strides,
    size_t num_dims,
    void* data
);

/// Destroys a tensor and frees any owned memory. Sets *tensor to NULL.
PTENSOR_API P10ErrorEnum p10_destroy(Ptensor* tensor);

/// Returns the total number of elements in the tensor.
PTENSOR_API size_t p10_get_size(Ptensor tensor);

/// Returns the total size of the tensor data in bytes.
PTENSOR_API size_t p10_get_size_bytes(Ptensor tensor);

/// Returns the data type of the tensor.
PTENSOR_API P10DTypeEnum p10_get_dtype(Ptensor tensor);

/// Fills shape with up to num_dims dimension sizes.
PTENSOR_API P10ErrorEnum p10_get_shape(Ptensor tensor, int64_t* shape, size_t num_dims);

/// Fills strides with up to num_dims per-element strides.
PTENSOR_API P10ErrorEnum p10_get_stride(Ptensor tensor, int64_t* strides, size_t num_dims);

/// Returns the number of dimensions of the tensor.
PTENSOR_API size_t p10_get_ndim(Ptensor tensor);

/// Returns a pointer to the raw tensor data.
PTENSOR_API void* p10_get_data(Ptensor tensor);

/// Returns 1 if the tensor has no elements, 0 otherwise.
PTENSOR_API int p10_is_empty(Ptensor tensor);

#ifdef __cplusplus
}
#endif

#endif

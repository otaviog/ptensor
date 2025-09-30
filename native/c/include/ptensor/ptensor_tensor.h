#ifndef PTENSOR_TENSOR_H_
#define PTENSOR_TENSOR_H_

#include <cstdint>

#include <stdint.h>

#include "config.h"
#include "ptensor_dtype.h"
#include "ptensor_error.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void* P10Tensor;

PTENSOR_API P10ErrorEnum p10_tensor_from_data(
    P10Tensor* tensor,
    P10DTypeEnum dtype,
    int64_t* shape,
    size_t num_dims,
    uint8_t* data
);

PTENSOR_API P10ErrorEnum p10_tensor_destroy(P10Tensor* tensor);

PTENSOR_API size_t p10_tensor_get_size(P10Tensor tensor);

PTENSOR_API P10DTypeEnum p10_tensor_get_dtype(P10Tensor tensor);

PTENSOR_API P10ErrorEnum
p10_tensor_get_shape(P10Tensor tensor, int64_t* shape, size_t num_dims);

PTENSOR_API size_t p10_tensor_get_dimensions(P10Tensor tensor);

PTENSOR_API void* p10_tensor_get_data(P10Tensor tensor);

#ifdef __cplusplus
}
#endif

#endif

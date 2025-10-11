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

typedef void* Ptensor;

PTENSOR_API P10ErrorEnum p10_from_data(
    Ptensor* tensor,
    P10DTypeEnum dtype,
    int64_t* shape,
    size_t num_dims,
    uint8_t* data
);

PTENSOR_API P10ErrorEnum p10_destroy(Ptensor* tensor);

PTENSOR_API size_t p10_get_size(Ptensor tensor);

PTENSOR_API P10DTypeEnum p10_get_dtype(Ptensor tensor);

PTENSOR_API P10ErrorEnum
p10_get_shape(Ptensor tensor, int64_t* shape, size_t num_dims);

PTENSOR_API size_t p10_get_dimensions(Ptensor tensor);

PTENSOR_API void* p10_get_data(Ptensor tensor);

#ifdef __cplusplus
}
#endif

#endif

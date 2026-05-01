#include "ptensor_tensor.h"

#include <ptensor/tensor.hpp>

#include "dtype_wrapper.hpp"
#include "ptensor/shape.hpp"
#include "ptensor/stride.hpp"
#include "tensor_wrapper.hpp"
#include "update_error_state.hpp"

PTENSOR_API P10ErrorEnum p10_from_data(
    Ptensor* tensor,
    P10DTypeEnum dtype,
    const int64_t* shape,
    size_t num_dims,
    void* data
) {
    auto shape_res = p10::make_shape(std::span(shape, num_dims));
    if (!shape_res.is_ok()) {
        return p10::update_error_state(shape_res.unwrap_err());
    }
    auto dtype_res = p10::wrap(dtype);
    if (!dtype_res.is_ok()) {
        return p10::update_error_state(dtype_res.unwrap_err());
    }

    auto options = p10::TensorOptions().dtype(dtype_res.unwrap());
    *tensor = wrap(std::move(p10::Tensor::from_data(data, shape_res.unwrap(), options)));

    return P10ErrorEnum::P10_OK;
}

PTENSOR_API P10ErrorEnum p10_from_data_strided(
    Ptensor* tensor,
    P10DTypeEnum dtype,
    const int64_t* shape,
    const int64_t* strides,
    size_t num_dims,
    void* data
) {
    auto shape_res = p10::make_shape(std::span(shape, num_dims));
    if (!shape_res.is_ok()) {
        return p10::update_error_state(shape_res.unwrap_err());
    }
    auto stride_res = p10::make_stride(std::span(strides, num_dims));
    if (!stride_res.is_ok()) {
        return p10::update_error_state(stride_res.unwrap_err());
    }
    auto dtype_res = p10::wrap(dtype);
    if (!dtype_res.is_ok()) {
        return p10::update_error_state(dtype_res.unwrap_err());
    }

    auto options = p10::TensorOptions().dtype(dtype_res.unwrap()).stride(stride_res.unwrap());
    *tensor = wrap(std::move(p10::Tensor::from_data(data, shape_res.unwrap(), options)));

    return P10ErrorEnum::P10_OK;
}

PTENSOR_API P10ErrorEnum p10_destroy(Ptensor* tensor) {
    if (tensor == nullptr) {
        return P10ErrorEnum::P10_OK;
    }
    auto* cxx_tensor = p10::unwrap(*tensor);
    delete cxx_tensor;
    *tensor = nullptr;
    return P10ErrorEnum::P10_OK;
}

PTENSOR_API size_t p10_get_size(Ptensor tensor) {
    return p10::unwrap(tensor)->size();
}

PTENSOR_API size_t p10_get_size_bytes(Ptensor tensor) {
    return p10::unwrap(tensor)->size_bytes();
}

PTENSOR_API P10DTypeEnum p10_get_dtype(Ptensor tensor) {
    return static_cast<P10DTypeEnum>(p10::unwrap(tensor)->dtype().value);
}

PTENSOR_API P10ErrorEnum p10_get_shape(Ptensor tensor, int64_t* shape, size_t num_dims) {
    const auto* cxx_tensor = p10::unwrap(tensor);
    auto shape_span = cxx_tensor->shape().as_span();
    for (size_t i = 0; i < std::min(num_dims, shape_span.size()); ++i) {
        shape[i] = shape_span[i];
    }
    return P10ErrorEnum::P10_OK;
}

PTENSOR_API P10ErrorEnum p10_get_stride(Ptensor tensor, int64_t* strides, size_t num_dims) {
    const auto* cxx_tensor = p10::unwrap(tensor);
    auto stride_span = cxx_tensor->stride().as_span();
    for (size_t i = 0; i < std::min(num_dims, stride_span.size()); ++i) {
        strides[i] = stride_span[i];
    }
    return P10ErrorEnum::P10_OK;
}

PTENSOR_API size_t p10_get_ndim(Ptensor tensor) {
    return p10::unwrap(tensor)->dims();
}

PTENSOR_API void* p10_get_data(Ptensor tensor) {
    return p10::unwrap(tensor)->as_bytes().data();
}

PTENSOR_API int p10_is_empty(Ptensor tensor) {
    return p10::unwrap(tensor)->empty() ? 1 : 0;
}

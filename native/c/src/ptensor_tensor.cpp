#include "ptensor_tensor.h"

#include <ptensor/tensor.hpp>

#include "dtype_wrapper.hpp"
#include "ptensor/shape.hpp"
#include "tensor_wrapper.hpp"
#include "update_error_state.hpp"

PTENSOR_API P10ErrorEnum p10_from_data(
    Ptensor* tensor,
    P10DTypeEnum dtype,
    int64_t* shape,
    size_t num_dims,
    uint8_t* data
) {
    auto shape_res = p10::make_shape(std::span(shape, num_dims));
    if (!shape_res.is_ok()) {
        return p10::update_error_state(shape_res.unwrap_err());
    }
    auto dtype_res = p10::wrap(dtype);
    if (!dtype_res.is_ok()) {
        return p10::update_error_state(dtype_res.unwrap_err());
    }

    *tensor = wrap(std::move(p10::Tensor::from_data(data, shape_res.unwrap(), dtype_res.unwrap())));

    return P10ErrorEnum::P10_OK;
}

PTENSOR_API P10ErrorEnum p10_destroy(Ptensor* tensor) {
    if (tensor == nullptr) {
        return P10ErrorEnum::P10_OK;
    }
    auto* cxx_tensor = unwrap(*tensor);
    delete cxx_tensor;
    *tensor = nullptr;
    return P10ErrorEnum::P10_OK;
}

PTENSOR_API size_t p10_get_size(Ptensor tensor) {
    const auto* cxx_tensor = unwrap(tensor);
    return cxx_tensor->size();
}

PTENSOR_API P10DTypeEnum p10_get_dtype(Ptensor tensor) {
    const auto* cxx_tensor = unwrap(tensor);
    return static_cast<P10DTypeEnum>(cxx_tensor->dtype().value);
}

PTENSOR_API P10ErrorEnum p10_get_shape(Ptensor tensor, int64_t* shape, size_t num_dims) {
    const auto* cxx_tensor = unwrap(tensor);
    auto shape_vec = cxx_tensor->shape().as_span();
    for (size_t i = 0; i < std::min(num_dims, shape_vec.size()); ++i) {
        shape[i] = shape_vec[i];
    }
    return P10ErrorEnum::P10_OK;
}

PTENSOR_API size_t p10_get_dimensions(Ptensor tensor) {
    return unwrap(tensor)->dims();
}

PTENSOR_API void* p10_get_data(Ptensor tensor) {
    return unwrap(tensor)->as_bytes().data();
}

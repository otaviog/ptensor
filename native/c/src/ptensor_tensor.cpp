#include "ptensor_tensor.h"

#include <ptensor/tensor.hpp>

#include "tensor_cast.hpp"
#include "update_error_state.hpp"

///////
// Tensor
namespace {
inline p10::DType to_cxx_dtype(P10DTypeEnum dtype) {
    return p10::DType(static_cast<p10::DType::Code>(dtype));
}
}  // namespace

PTENSOR_API P10ErrorEnum p10_tensor_from_data(
    P10Tensor* tensor,
    P10DTypeEnum dtype,
    int64_t* shape,
    size_t num_dims,
    uint8_t* data
) {
    std::vector<int64_t> shape_vec(shape, shape + num_dims);

    *tensor = to_tensor_c(new p10::Tensor(to_cxx_dtype(dtype), shape_vec, data));

    return P10ErrorEnum::P10_OK;
}

PTENSOR_API P10ErrorEnum p10_tensor_destroy(P10Tensor* tensor) {
    if (tensor == nullptr) {
        return P10ErrorEnum::P10_OK;
    }
    auto* cxx_tensor = to_tensor_cxx(*tensor);
    delete cxx_tensor;
    *tensor = nullptr;
    return P10ErrorEnum::P10_OK;
}

PTENSOR_API size_t p10_tensor_get_size(P10Tensor tensor) {
    const auto* cxx_tensor = to_tensor_cxx(tensor);
    return cxx_tensor->size();
}

PTENSOR_API P10DTypeEnum p10_tensor_get_dtype(P10Tensor tensor) {
    const auto* cxx_tensor = to_tensor_cxx(tensor);
    return static_cast<P10DTypeEnum>(cxx_tensor->dtype().code());
}

PTENSOR_API P10ErrorEnum p10_tensor_get_shape(P10Tensor tensor, int64_t* shape, size_t num_dims) {
    const auto* cxx_tensor = to_tensor_cxx(tensor);
    auto shape_vec = cxx_tensor->shape();
    for (size_t i = 0; i < std::min(num_dims, shape_vec.size()); ++i) {
        shape[i] = shape_vec[i];
    }
    return P10ErrorEnum::P10_OK;
}

PTENSOR_API size_t p10_tensor_get_dimensions(P10Tensor tensor) {
    const auto* cxx_tensor = to_tensor_cxx(tensor);
    return cxx_tensor->shape().size();
}

PTENSOR_API void* p10_tensor_get_data(P10Tensor tensor) {
    auto* cxx_tensor = to_tensor_cxx(tensor);
    return cxx_tensor->data<void>();
}

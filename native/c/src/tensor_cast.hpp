#pragma once

#include <ptensor/tensor.hpp>
#include "ptensor_tensor.h"

inline p10::Tensor* to_tensor_cxx(P10Tensor tensor) {
    return reinterpret_cast<p10::Tensor*>(tensor);
}

inline P10Tensor to_tensor_c(p10::Tensor* ptr) {
    return reinterpret_cast<P10Tensor>(ptr);
}
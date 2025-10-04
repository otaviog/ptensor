#pragma once

#include <ptensor/tensor.hpp>

#include "ptensor_tensor.h"

inline p10::Tensor* unwrap(P10Tensor tensor) {
    return reinterpret_cast<p10::Tensor*>(tensor);
}

inline P10Tensor wrap(p10::Tensor* ptr) {
    return reinterpret_cast<P10Tensor>(ptr);
}

inline P10Tensor wrap(p10::Tensor&& tensor) {
    return reinterpret_cast<P10Tensor>(new p10::Tensor(std::move(tensor)));
}

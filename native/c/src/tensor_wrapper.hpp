#pragma once

#include <ptensor/tensor.hpp>

#include "ptensor_tensor.h"

inline p10::Tensor* unwrap(Ptensor tensor) {
    return reinterpret_cast<p10::Tensor*>(tensor);
}

inline Ptensor wrap(p10::Tensor* ptr) {
    return reinterpret_cast<Ptensor>(ptr);
}

inline Ptensor wrap(p10::Tensor&& tensor) {
    return reinterpret_cast<Ptensor>(new p10::Tensor(std::move(tensor)));
}

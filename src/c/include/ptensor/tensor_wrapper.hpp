#pragma once

#include <ptensor/tensor.hpp>

#include "ptensor_tensor.h"

namespace p10 {
inline Tensor* unwrap(Ptensor tensor) {
    assert(tensor != nullptr && "Null Ptensor handle");
    return reinterpret_cast<Tensor*>(tensor);
}

inline Tensor& unwrap_ref(Ptensor tensor) {
    assert(tensor != nullptr && "Null Ptensor handle");
    return *reinterpret_cast<Tensor*>(tensor);
}

inline const Tensor& unwrap_ref_const(Ptensor tensor) {
    assert(tensor != nullptr && "Null Ptensor handle");
    return *reinterpret_cast<Tensor*>(tensor);
}

inline Ptensor wrap(Tensor* ptr) {
    return reinterpret_cast<Ptensor>(ptr);
}

inline Ptensor wrap(Tensor&& tensor) {
    return reinterpret_cast<Ptensor>(new p10::Tensor(std::move(tensor)));
}
}  // namespace p10

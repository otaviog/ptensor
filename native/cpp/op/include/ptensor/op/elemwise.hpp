#pragma once

#include "ptensor/ptensor_error.hpp"

namespace p10 {
class Tensor;
}

namespace p10::op {
PtensorError add_elemwise(const Tensor& a, const Tensor& b, Tensor& out);
PtensorError subtract_elemwise(const Tensor& a, const Tensor& b, Tensor& out);
}  // namespace p10::tensorop
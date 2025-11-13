#pragma once

#include "ptensor/p10_error.hpp"

namespace p10 {
class Tensor;
}

namespace p10::op {
P10Error add_elemwise(const Tensor& a, const Tensor& b, Tensor& out);
P10Error subtract_elemwise(const Tensor& a, const Tensor& b, Tensor& out);
P10Error multiply_elemwise(const Tensor& a, const Tensor& b, Tensor& out);
}  // namespace p10::op
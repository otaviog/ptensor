#pragma once

#include "ptensor/ptensor_error.hpp"

namespace p10 {
class Tensor;
}

namespace p10::op {
PtensorError resize(const Tensor& input, Tensor& output, size_t new_width, size_t new_height);
}  // namespace p10::op
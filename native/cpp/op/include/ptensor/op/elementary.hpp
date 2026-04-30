#pragma once

#include "ptensor/p10_error.hpp"

namespace p10 {
class Tensor;
}

namespace p10::op {
P10Error subtract_elements(Tensor& a, double value);
}  // namespace p10::op

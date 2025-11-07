#pragma once
#include "ptensor/p10_error.hpp"

namespace p10 {
class Tensor;
}

namespace p10::op {
void multiply_scalar(Tensor& a, double scalar);
}

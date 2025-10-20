#pragma once

#include <ptensor/p10_error.hpp>

namespace p10 {
class Tensor;
}

namespace p10::op {
P10Error image_to_tensor(const Tensor& image_tensor, Tensor& out_float_tensor);

P10Error image_from_tensor(const Tensor& float_tensor, Tensor& out_image_tensor);
}  // namespace p10::op

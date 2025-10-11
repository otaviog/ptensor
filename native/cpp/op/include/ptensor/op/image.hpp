#pragma once

#include <ptensor/ptensor_error.hpp>

namespace p10 {
class Tensor;
}

namespace p10::op {
PtensorError image_to_tensor(const Tensor& image_tensor, Tensor& out_float_tensor);

PtensorError image_from_tensor(const Tensor& float_tensor, Tensor& out_image_tensor);
}  // namespace p10::op

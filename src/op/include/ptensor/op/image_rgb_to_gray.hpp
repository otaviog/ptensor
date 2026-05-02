#pragma once

#include <ptensor/p10_error.hpp>

namespace p10 {
class Tensor;
}

namespace p10::op {
p10::P10Error image_rgb_to_gray(const p10::Tensor& rgb_image, p10::Tensor& gray_image);
}

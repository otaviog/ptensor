#pragma once

#include <ptensor/p10_error.hpp>

namespace p10 {
class Tensor;
};

namespace p10::op {
P10Error crop(const Tensor &image, size_t x, int y, size_t w, size_t h, Tensor &crop);
}

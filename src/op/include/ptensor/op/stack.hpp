#pragma once
#include <span>

#include <ptensor/p10_error.hpp>

namespace p10 {
class Tensor;
}

namespace p10::op {
p10::P10Error stack(std::span<const p10::Tensor> inputs, int64_t axis, p10::Tensor& output);
}

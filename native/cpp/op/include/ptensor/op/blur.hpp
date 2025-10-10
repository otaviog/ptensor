#pragma once

#include <array>
#include <span>

#include "ptensor/ptensor_result.hpp"

namespace p10 {
class Tensor;
}

namespace p10::op {
class GaussianBlur {
  public:
    static constexpr size_t MAX_KERNEL_SIZE = 25;

    static PtensorResult<GaussianBlur> create(size_t kernel_size, float sigma);

    PtensorError operator()(const Tensor& input, Tensor& output) const;

  private:
    using KernelStorage = std::array<float, MAX_KERNEL_SIZE>;

    GaussianBlur(KernelStorage kernel, size_t kernel_size) :
        kernel_data_(kernel),
        kernel_size_(kernel_size) {}

    std::span<const float> get_kernel() const {
        return std::span<const float>(kernel_data_.data(), kernel_size_);
    }

    KernelStorage kernel_data_;
    size_t kernel_size_;
};

}  // namespace p10::op

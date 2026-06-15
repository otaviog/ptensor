#pragma once

#include <array>
#include <span>

#include <ptensor/p10_result.hpp>
#include <ptensor/tensor.hpp>

namespace p10::op {
class GaussianBlur {
  public:
    static constexpr size_t MAX_KERNEL_SIZE = 25;

    static P10Result<GaussianBlur> create(size_t kernel_size, float sigma);

    P10Error transform(const Tensor& input, Tensor& output);

  private:
    struct {
        std::array<float, MAX_KERNEL_SIZE> data;
        size_t size;

        std::span<const float> get() const {
            return std::span<const float>(data.data(), size);
        }

        std::span<float> get() {
            return std::span<float>(data.data(), size);
        }
        
    } kernel_;

    GaussianBlur(size_t kernel_size) :
        kernel_{.data = {}, .size = kernel_size} {}

    Tensor horizontal_out_;
};

}  // namespace p10::op

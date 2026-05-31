#pragma once

#include <ptensor/p10_error.hpp>
#include <ptensor/tensor.hpp>

namespace p10::recog {
class BfPreprocessing {
  public:
    BfPreprocessing(size_t target_size) : target_size_(target_size) {}

    P10Result<float> process(Tensor& images, Tensor& preprocessed);

  private:
    size_t target_size_;
    Tensor resize_buffer_;
};
}  // namespace p10::recog

#pragma once

#include <ptensor/p10_error.hpp>
#include <ptensor/tensor.hpp>

namespace p10::recog {
class BfPreprocessing {
  public:
    P10Result<float> process(Tensor& images, Tensor& preprocessed);

  private:    
    size_t target_size_;
    Tensor resize_buffer_;
    Tensor float_buffer_;
};
}  // namespace p10::recog

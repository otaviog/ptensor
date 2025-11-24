#pragma once

#include <memory>

#include <ptensor/dtype.hpp>
#include <ptensor/p10_error.hpp>

namespace p10 {
class Tensor;
}

namespace p10::op {
class WindowFunction {
  public:
    enum Function { Hanning, Hamming, Identity };

    WindowFunction(Function func);

    P10Error transform(const Tensor& input, Tensor& output);

    P10Error transform_borders(const Tensor& input, Tensor& output, size_t border_size);

  private:
    P10Error generate_window(size_t size, Dtype type);

    Function func_;
    std::unique_ptr<Tensor> window_;
};

}  // namespace p10::op
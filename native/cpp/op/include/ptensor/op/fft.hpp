#pragma once

#include <kissfft/kissfft.hh>

namespace p10 {
class Tensor;
}

namespace p10::op {

class FFT {
  public:
    FFT(size_t nfft, bool inverse);
    void forward(const Tensor& time, Tensor& frequency) const;
    void inverse(const Tensor& input, Tensor& output) const;

  private:
    kissfft<float> kiss_;
};
}  // namespace p10::tensorop
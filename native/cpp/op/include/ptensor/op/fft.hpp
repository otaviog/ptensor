#pragma once

#include <cstddef>

#include "ptensor/p10_error.hpp"

namespace p10 {
class Tensor;
}

namespace p10::op {

class FFT {
  public:
  enum FFTType { Forward = false, Inverse = true };
    FFT(size_t nfft, FFTType type);
    ~FFT();

    FFT(const FFT&) = delete;
    FFT& operator=(const FFT&) = delete;

    P10Error transform(const Tensor& input, Tensor& output) const;

  private:
    P10Error forward(const Tensor& time, Tensor& frequency) const;
    P10Error inverse(const Tensor& input, Tensor& output) const;
    void* kiss_ = nullptr;
    FFTType type_ = FFTType::Forward;
};
}  // namespace p10::op
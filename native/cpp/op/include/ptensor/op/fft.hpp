#pragma once


namespace p10 {
class Tensor;
}

namespace p10::op {

class FFT {
  public:
    FFT(size_t nfft, bool inverse);
    ~FFT();
    
    FFT(const FFT&) = delete;
    FFT& operator=(const FFT&) = delete;


    void forward(const Tensor& time, Tensor& frequency) const;
    void inverse(const Tensor& input, Tensor& output) const;

  private:
    void* kiss_ = nullptr;
};
}  // namespace p10::op
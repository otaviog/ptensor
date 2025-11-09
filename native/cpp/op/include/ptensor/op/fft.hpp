#pragma once

#include <vector>

#include <ptensor/p10_error.hpp>
#include <ptensor/tensor.hpp>

namespace p10 {
class Tensor;
}

namespace p10::op {
class FftOptions;

/**
 * @brief Fft is NOT thread-safe. Do not use the same instance concurrently from multiple threads.
 */
class Fft {
  public:
    enum Normalize { None, ByN, BySqrtN };

    enum Direction { Forward = 0, ForwardReal = 1, Inverse = 2, InverseReal = 3 };

    Fft(size_t nfft, const FftOptions& options);
    ~Fft();

    Fft(const Fft&) = delete;
    Fft& operator=(const Fft&) = delete;

    P10Error transform(const Tensor& input, Tensor& output) const;

  private:
    P10Error forward(const Tensor& time, Tensor& frequency) const;
    P10Error forward_real(const Tensor& signal_in, Tensor& freq_out) const;
    P10Error inverse(const Tensor& input, Tensor& output) const;
    P10Error inverse_real(const Tensor& frequency, Tensor& time) const;

    void* kiss_ = nullptr;
    Direction direction_ = Direction::Forward;
    Normalize normalize_ = Normalize::None;
    size_t nfft_ = 0;

    // Not thread-safe: buffer_ is modified by const methods.
    mutable std::vector<Tensor> buffer_;
};

class FftOptions {
  public:
    FftOptions(Fft::Direction direction = Fft::Forward, Fft::Normalize normalize = Fft::None) :
        direction_(direction),
        normalize_(normalize) {}

    Fft::Direction direction() const {
        return direction_;
    }

    Fft::Normalize normalize() const {
        return normalize_;
    }

    FftOptions& direction(Fft::Direction t) {
        direction_ = t;
        return *this;
    }

    FftOptions& normalize(Fft::Normalize n) {
        normalize_ = n;
        return *this;
    }

  private:
    Fft::Direction direction_;
    Fft::Normalize normalize_;
};

}  // namespace p10::op
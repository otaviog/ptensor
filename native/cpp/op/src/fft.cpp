#include "fft.hpp"

#include "tensor.hpp"

namespace p10::op {

FFT::FFT(size_t nfft, bool inverse) : kiss_(nfft, inverse) {}

void FFT::forward(const Tensor& time, Tensor& frequency) const {
    // validate_time_argument(time);
    const auto num_signals = time.shape(0);
    const auto signal_size = time.shape(1);
    frequency.create(DType::FLOAT32, {num_signals, 2, signal_size});

    // kiss_.transform(time.data<float>())
}

void FFT::inverse(const Tensor& frequency, Tensor& time) const {}

namespace {}  // namespace

}  // namespace p10::op

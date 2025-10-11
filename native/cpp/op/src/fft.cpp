#include "fft.hpp"
#include <kissfft/kissfft.hh>

#include "tensor.hpp"

namespace p10::op {
namespace {
    inline kissfft<float>* unwrap(void* ptr) {
        return reinterpret_cast<kissfft<float>*>(ptr);
    }

    inline const kissfft<float>* unwrap(const void* ptr) {
        return reinterpret_cast<const kissfft<float>*>(ptr);
    }

    inline void* wrap(kissfft<float>* ptr) {
        return reinterpret_cast<void*>(ptr);
    }
}

FFT::FFT(size_t nfft, bool inverse) : kiss_(nfft, inverse) {
    kiss_ = wrap(new kissfft<float>(nfft, inverse));

}
FFT::~FFT() {
    delete unwrap(kiss_);
}

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

#include "fft.hpp"

#include <kissfft.hh>
#include <ptensor/tensor.hpp>

#include "ptensor/p10_error.hpp"
#include "ptensor/p10_result.hpp"

namespace p10::op {
namespace {
    inline kissfft<float>* unwrap(void* ptr);
    inline void* wrap(kissfft<float>* ptr);
    P10Result<std::tuple<int64_t, int64_t>> validate_time_arguments(const Tensor& time);
    P10Result<std::tuple<int64_t, int64_t>> validate_frequency_arguments(const Tensor& frequency);
}  // namespace

FFT::FFT(size_t nfft, bool inverse) {
    kiss_ = wrap(new kissfft<float>(nfft, inverse));
    inverse_ = inverse;
}

FFT::~FFT() {
    delete unwrap(kiss_);
}

P10Error FFT::transform(const Tensor& input, Tensor& output) const {
    if (inverse_) {
        return inverse(input, output);
    } else {
        return forward(input, output);
    }
}

P10Error FFT::forward(const Tensor& time, Tensor& frequency) const {
    auto validation = validate_time_arguments(time);
    if (validation.is_error()) {
        return validation.err();
    }

    const auto [num_signals, signal_size] = validation.unwrap();
    frequency.create(
        make_shape(num_signals, signal_size / 2 + 1, 2),
        TensorOptions().dtype(Dtype::Float32)
    );

    unwrap(kiss_)->transform_real(
        time.as_span1d<float>().unwrap().data(),
        frequency.as_span1d<std::complex<float>>().unwrap().data()
    );
    return P10Error::Ok;
}

P10Error FFT::inverse(const Tensor& frequency, Tensor& time) const {
    auto validation = validate_frequency_arguments(frequency);
    if (validation.is_error()) {
        return validation.err();
    }

    const auto [num_signals, signal_size] = validation.unwrap();
    time.create(make_shape(num_signals, signal_size, 2), TensorOptions().dtype(Dtype::Float32));
    unwrap(kiss_)->transform(
        frequency.as_span1d<std::complex<float>>().unwrap().data(),
        time.as_span1d<std::complex<float>>().unwrap().data()
    );
    return P10Error::Ok;
}

namespace {
    inline kissfft<float>* unwrap(void* ptr) {
        return reinterpret_cast<kissfft<float>*>(ptr);
    }

    inline void* wrap(kissfft<float>* ptr) {
        return reinterpret_cast<void*>(ptr);
    }

    P10Result<std::tuple<int64_t, int64_t>> validate_time_arguments(const Tensor& time) {
        if (time.dims() != 2) {
            return Err(P10Error::InvalidArgument << "Input tensor must have 2 dimensions");
        }
        if (time.dtype() != Dtype::Float32) {
            return Err(P10Error::InvalidArgument << "Input tensor must have Float32 dtype");
        }
        const auto num_signals = time.shape(0).unwrap();
        const auto signal_size = time.shape(1).unwrap();
        if (signal_size % 2 != 0) {
            return Err(P10Error::InvalidArgument << "Input tensor signal size must be even");
        }
        return Ok(std::make_tuple(num_signals, signal_size));
    }

    P10Result<std::tuple<int64_t, int64_t>> validate_frequency_arguments(const Tensor& frequency) {
        if (frequency.dims() != 3) {
            return Err(P10Error::InvalidArgument << "Input tensor must have 3 dimensions");
        }
        if (frequency.dtype() != Dtype::Float32) {
            return Err(P10Error::InvalidArgument << "Input tensor must have Float32 dtype");
        }
        const auto num_signals = frequency.shape(0).unwrap();
        const auto signal_size = (frequency.shape(1).unwrap() - 1) * 2;
        if (frequency.shape(2).unwrap() != 2) {
            return Err(
                P10Error::InvalidArgument
                << "Input tensor last dimension must be 2 (real and imaginary)"
            );
        }
        return Ok(std::make_tuple(num_signals, signal_size));
    }

}  // namespace
}  // namespace p10::op

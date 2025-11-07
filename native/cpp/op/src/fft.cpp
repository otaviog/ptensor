#include "fft.hpp"

#include <kissfft.hh>
#include <ptensor/tensor.hpp>

#include "ptensor/p10_error.hpp"
#include "ptensor/p10_result.hpp"
#include "tensor_scalar.hpp"

namespace p10::op {
namespace {
    inline kissfft<float>* unwrap(void* ptr);
    inline void* wrap(kissfft<float>* ptr);
    P10Result<std::tuple<int64_t, int64_t, bool>> validate_time_arguments(const Tensor& time);
    P10Result<std::tuple<int64_t, int64_t>> validate_frequency_arguments(const Tensor& frequency);
    p10::P10Error normalize_time_tensor(Tensor& complex_time, size_t signal_size);
}  // namespace

Fft::Fft(size_t nfft, const FftOptions& options) :
    direction_(options.direction()),
    normalize_(options.normalize()) {
    kiss_ = wrap(new kissfft<float>(nfft, direction_ == Direction::Inverse));
}

Fft::~Fft() {
    delete unwrap(kiss_);
}

P10Error Fft::transform(const Tensor& input, Tensor& output) const {
    if (direction_ == Direction::Forward) {
        return forward(input, output);
    } else {
        return inverse(input, output);
    }
}

P10Error Fft::forward(const Tensor& time, Tensor& frequency) const {
    auto validation = validate_time_arguments(time);
    if (validation.is_error()) {
        return validation.err();
    }

    const auto [num_signals, signal_size, is_complex_input] = validation.unwrap();
    frequency.create(
        make_shape(num_signals, signal_size / 2 + 1, 2),
        TensorOptions().dtype(Dtype::Float32)
    );

    if (is_complex_input) {
        unwrap(kiss_)->transform(
            time.as_span1d<std::complex<float>>().unwrap().data(),
            frequency.as_span1d<std::complex<float>>().unwrap().data()
        );
    } else {
        unwrap(kiss_)->transform_real(
            time.as_span1d<float>().unwrap().data(),
            frequency.as_span1d<std::complex<float>>().unwrap().data()
        );
    }

    return P10Error::Ok;
}

P10Error Fft::inverse(const Tensor& frequency, Tensor& time) const {
    auto validation = validate_frequency_arguments(frequency);
    if (validation.is_error()) {
        return validation.err();
    }

    const auto [num_signals, signal_size] = validation.unwrap();

    if (auto err = time.create(
            make_shape(num_signals, signal_size, 2),
            TensorOptions().dtype(Dtype::Float32)
        );
        err.is_error()) {
        return err;
    }
    unwrap(kiss_)->transform(
        frequency.as_span1d<std::complex<float>>().unwrap().data(),
        time.as_span1d<std::complex<float>>().unwrap().data()
    );

    if (normalize_ == Normalize::ByN) {
        return normalize_time_tensor(time, signal_size);
    }

    return P10Error::Ok;
}

namespace {
    inline kissfft<float>* unwrap(void* ptr) {
        return reinterpret_cast<kissfft<float>*>(ptr);
    }

    inline void* wrap(kissfft<float>* ptr) {
        return reinterpret_cast<void*>(ptr);
    }

    P10Result<std::tuple<int64_t, int64_t, bool>> validate_time_arguments(const Tensor& time) {
        if (time.dims() != 2 && time.dims() != 3) {
            return Err(
                P10Error::InvalidArgument
                << "Input tensor must have 2[N x T] or 3 dimensions [N x T x 2]"
            );
        }
        if (time.dtype() != Dtype::Float32) {
            return Err(P10Error::InvalidArgument << "Input tensor must have Float32 dtype");
        }

        if (time.device() != Device::Cpu) {
            return Err(P10Error::InvalidArgument << "Input tensor must be on CPU device");
        }

        const auto num_signals = time.shape(0).unwrap();
        const auto signal_size = time.shape(1).unwrap();
        if (signal_size % 2 != 0) {
            return Err(P10Error::InvalidArgument << "Input tensor signal size must be even");
        }
        const bool is_complex = (time.dims() == 3);
        if (is_complex && time.shape(2).unwrap() != 2) {
            return Err(
                P10Error::InvalidArgument
                << "Input tensor last dimension must be 2 (real and imaginary) if complex"
            );
        }
        return Ok(std::make_tuple(num_signals, signal_size, is_complex));
    }

    P10Result<std::tuple<int64_t, int64_t>> validate_frequency_arguments(const Tensor& frequency) {
        if (frequency.dims() != 3) {
            return Err(
                P10Error::InvalidArgument << "Input tensor must have 3 dimensions [N x F x 2]"
            );
        }
        if (frequency.dtype() != Dtype::Float32) {
            return Err(P10Error::InvalidArgument << "Input tensor must have Float32 dtype");
        }
        if (frequency.device() != Device::Cpu) {
            return Err(P10Error::InvalidArgument << "Input tensor must be on CPU device");
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

    p10::P10Error normalize_time_tensor(Tensor& complex_time, size_t signal_size) {
        assert(complex_time.dims() == 3);
        const auto num_signals = complex_time.shape(0).unwrap();
        const auto divider = 1.0f / static_cast<float>(signal_size);
        auto time_span = complex_time.as_planar_span3d<float>().unwrap();
        for (int64_t n = 0; n < num_signals; n++) {
            auto signal_span = time_span.plane(n);
            for (size_t t = 0; t < signal_size; t++) {
                signal_span[t] = signal_span[t * 2] * divider;
            }
        }
        return complex_time.create(
            make_shape(num_signals, signal_size),
            TensorOptions().dtype(Dtype::Float32)
        );
    }
}  // namespace
}  // namespace p10::op

#include "fft.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <kissfft.hh>
#include <ptensor/tensor.hpp>
#include <ptensor/tensor_print.hpp>

#include "ptensor/p10_error.hpp"
#include "ptensor/p10_result.hpp"
#include "tensor_scalar.hpp"

namespace p10::op {
namespace {
    constexpr size_t BUFFER_INVERSE_REAL_FREQ_CONJUGATE = 0;
    constexpr size_t BUFFER_INVERSE_REAL_SIGNAL_COMPLEX = 1;
    constexpr size_t BUFFER_INVERSE_REAL_COUNT = 2;

    constexpr size_t BUFFER_FORWARD_REAL_SIGNAL_COMPLEX = 0;
    constexpr size_t BUFFER_FORWARD_REAL_FREQUENCY_FULL = 1;
    constexpr size_t BUFFER_FORWARD_COUNT = 2;

    inline kissfft<float>* unwrap(void* ptr);
    inline void* wrap(kissfft<float>* ptr);
    P10Result<int64_t> validate_frequency_arguments(const Tensor& frequency, size_t nfft);
    P10Error check_input_device_and_dtype(const Tensor& tensor);
    void
    normalize_complex_signal(const std::span<std::complex<float>>& signal_span, float* signal_out);

}  // namespace

Fft::Fft(size_t nfft, const FftOptions& options) :
    direction_(options.direction()),
    normalize_(options.normalize()),
    nfft_(nfft) {
    size_t kiss_nfft = nfft;
    if (direction_ == Direction::ForwardReal || direction_ == Direction::InverseReal) {
        kiss_nfft = (nfft - 1) * 2;
    }

    kiss_ = wrap(new kissfft<float>(
        kiss_nfft,
        direction_ != Direction::Forward && direction_ != Direction::ForwardReal
    ));
}

Fft::~Fft() {
    delete unwrap(kiss_);
}

P10Error Fft::transform(const Tensor& input, Tensor& output) const {
    switch (direction_) {
        case Direction::Forward:
            return forward(input, output);
        case Direction::ForwardReal:
            return forward_real(input, output);
        case Direction::Inverse:
            return inverse(input, output);
        case Direction::InverseReal:
            return inverse_real(input, output);
        default:
            return P10Error::InvalidArgument << "Invalid FFT direction specified";
    }
}

P10Error Fft::forward(const Tensor& signal_in, Tensor& freq_out) const {
    if (signal_in.dims() != 3) {
        return P10Error::InvalidArgument
            << "Input tensor must have 3 dimensions [N x T x 2] for complex FFT";
    }
    P10_RETURN_IF_ERROR(check_input_device_and_dtype(signal_in));

    const auto signal_size = signal_in.shape(1).unwrap();
    if (size_t(signal_size) != nfft_) {
        return P10Error::InvalidArgument
            << "Input tensor signal size does not match initialized FFT size";
    }

    const auto num_signals = signal_in.shape(0).unwrap();

    freq_out.create(make_shape(num_signals, nfft_, 2), TensorOptions().dtype(Dtype::Float32));
    auto freq_out_s = freq_out.as_span2d<std::complex<float>>().unwrap();

    const auto signal_in_s = signal_in.as_span2d<const std::complex<float>>().unwrap();
    for (auto signal_idx = 0; signal_idx < num_signals; signal_idx++) {
        unwrap(kiss_)->transform(signal_in_s.row(signal_idx), freq_out_s.row(signal_idx));
    }

    return P10Error::Ok;
}

P10Error Fft::forward_real(const Tensor& signal_in, Tensor& freq_out) const {
    buffer_.resize(BUFFER_FORWARD_COUNT);

    if (signal_in.dims() != 2) {
        return P10Error::InvalidArgument << "Input tensor must have [N x T]";
    }
    P10_RETURN_IF_ERROR(check_input_device_and_dtype(signal_in));

    const auto signal_size = signal_in.shape(1).unwrap();
    if (size_t(signal_size) != (nfft_ - 1) * 2) {
        return P10Error::InvalidArgument
            << "Input tensor signal size must match (nfft - 1) * 2 for real FFT";
    }

    const auto num_signals = signal_in.shape(0).unwrap();
    freq_out.create(make_shape(num_signals, nfft_, 2), TensorOptions().dtype(Dtype::Float32));
    auto freq_out_s = freq_out.as_span2d<std::complex<float>>().unwrap();

    Tensor& signal_cpx = buffer_[BUFFER_FORWARD_REAL_SIGNAL_COMPLEX];
    P10_RETURN_IF_ERROR(signal_cpx.create(make_shape(signal_size, 2), Dtype::Float32));
    Tensor& freq_full = buffer_[BUFFER_FORWARD_REAL_FREQUENCY_FULL];
    P10_RETURN_IF_ERROR(freq_full.create(make_shape(signal_size, 2), Dtype::Float32));

    const auto signal_in_s = signal_in.as_span2d<const float>().unwrap();
    auto signal_cpx_s = signal_cpx.as_span1d<std::complex<float>>().unwrap();
    auto freq_full_s = freq_full.as_span1d<std::complex<float>>().unwrap();

    for (auto signal_idx = 0; signal_idx < num_signals; signal_idx++) {
        const auto in_row = signal_in_s.row(signal_idx);
        std::transform(in_row, in_row + signal_size, signal_cpx_s.data(), [](float v) {
            return std::complex<float> {v, 0.0f};
        });

        unwrap(kiss_)->transform(signal_cpx_s.data(), freq_full_s.data());
        std::copy(freq_full_s.data(), freq_full_s.data() + nfft_, freq_out_s.row(signal_idx));
    }
    return P10Error::Ok;
}

P10Error Fft::inverse(const Tensor& freq_in, Tensor& signal_in) const {
    auto validation = validate_frequency_arguments(freq_in, nfft_);
    if (validation.is_error()) {
        return validation.err();
    }
    const auto num_signals = validation.unwrap();

    P10_RETURN_IF_ERROR(
        signal_in.create(make_shape(num_signals, nfft_, 2), TensorOptions().dtype(Dtype::Float32))
    );

    const auto freq_s = freq_in.as_span2d<const std::complex<float>>().unwrap();
    auto time_s = signal_in.as_span2d<std::complex<float>>().unwrap();
    for (int64_t signal_idx = 0; signal_idx < num_signals; signal_idx++) {
        unwrap(kiss_)->transform(freq_s.row(signal_idx), time_s.row(signal_idx));
    }

    if (normalize_ == Normalize::BySqrtN) {
        multiply_scalar(signal_in, 1.0 / std::sqrt(static_cast<double>(nfft_)));
    } else if (normalize_ == Normalize::ByN) {
        multiply_scalar(signal_in, 1.0 / static_cast<double>(nfft_));
    }

    return P10Error::Ok;
}

P10Error Fft::inverse_real(const Tensor& freq_in, Tensor& signal_out) const {
    auto validation = validate_frequency_arguments(freq_in, nfft_);
    if (validation.is_error()) {
        return validation.err();
    }
    const auto num_signals = validation.unwrap();
    const auto signal_size = (nfft_ - 1) * 2;

    if (auto err = signal_out.create(
            make_shape(num_signals, signal_size),
            TensorOptions().dtype(Dtype::Float32)
        );
        err.is_error()) {
        return err;
    }

    buffer_.resize(BUFFER_INVERSE_REAL_COUNT);
    Tensor& freq_conjs = buffer_[BUFFER_INVERSE_REAL_FREQ_CONJUGATE];
    Tensor& signal_cpx = buffer_[BUFFER_INVERSE_REAL_SIGNAL_COMPLEX];

    P10_RETURN_IF_ERROR(
        freq_conjs.create(make_shape(signal_size, 2), TensorOptions().dtype(Dtype::Float32))
    );
    P10_RETURN_IF_ERROR(
        signal_cpx.create(make_shape(signal_size, 2), TensorOptions().dtype(Dtype::Float32))
    );

    const auto freq_in_s = freq_in.as_span2d<const std::complex<float>>().unwrap();
    auto freq_conjs_s = freq_conjs.as_span1d<std::complex<float>>().unwrap();
    auto signal_cpx_s = signal_cpx.as_span1d<std::complex<float>>().unwrap();
    auto signal_out_s = signal_out.as_span2d<float>().unwrap();
    for (int64_t signal_idx = 0; signal_idx < num_signals; signal_idx++) {
        std::copy(freq_in_s.row(signal_idx), freq_in_s.row(signal_idx) + nfft_, freq_conjs_s.data());
        std::transform(
            freq_in_s.row(signal_idx) + 1,
            freq_in_s.row(signal_idx) + nfft_ - 1,
            freq_conjs_s.data() + nfft_,
            [](const std::complex<float>& c) { return std::conj(c); }
        );
        unwrap(kiss_)->transform(freq_conjs_s.data(), signal_cpx_s.data());
        normalize_complex_signal(signal_cpx_s, signal_out_s.row(signal_idx));
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

    P10Result<int64_t> validate_frequency_arguments(const Tensor& frequency, size_t nfft) {
        if (frequency.dims() != 3) {
            return Err(
                P10Error::InvalidArgument << "Input tensor must have 3 dimensions [N x F x 2]"
            );
        }

        if (size_t(frequency.shape(1).unwrap()) != nfft) {
            return Err(
                P10Error::InvalidArgument
                << "Input tensor frequency size does not match initialized FFT size"
            );
        }

        if (auto err = check_input_device_and_dtype(frequency); err != P10Error::Ok) {
            return Err(err);
        }

        int64_t num_signals = frequency.shape(0).unwrap();
        if (frequency.shape(2).unwrap() != 2) {
            return Err(
                P10Error::InvalidArgument
                << "Input tensor last dimension must be 2 (real and imaginary)"
            );
        }
        return Ok(std::move(num_signals));
    }

    P10Error check_input_device_and_dtype(const Tensor& tensor) {
        if (tensor.dtype() != Dtype::Float32) {
            return P10Error::InvalidArgument << "Input tensor must have Float32 dtype";
        }
        if (tensor.device() != Device::Cpu) {
            return P10Error::InvalidArgument << "Input tensor must be on CPU device";
        }

        return P10Error::Ok;
    }

    void
    normalize_complex_signal(const std::span<std::complex<float>>& signal_span, float* signal_out) {
        const size_t signal_size = signal_span.size();
        const auto divider = 1.0f / static_cast<float>(signal_size);
        for (size_t t = 0; t < signal_size; t++) {
            signal_out[t] = signal_span[t].real() * divider;
        }
    }
}  // namespace
}  // namespace p10::op

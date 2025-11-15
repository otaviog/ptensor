#include "fft.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <pocketfft_hdronly.h>
#include <ptensor/tensor.hpp>
#include <ptensor/tensor_print.hpp>

#include "ptensor/p10_error.hpp"
#include "ptensor/p10_result.hpp"
#include "tensor_scalar.hpp"

namespace p10::op {
namespace {

    P10Result<int64_t> validate_frequency_arguments(const Tensor& frequency);
    P10Error check_input_device_and_dtype(const Tensor& tensor);

}  // namespace

Fft::Fft(const FftOptions& options) :
    direction_(options.direction()),
    normalize_(options.normalize()) {}

Fft::~Fft() {}

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

P10Error Fft::forward(const Tensor&, Tensor&) const {
    return P10Error::NotImplemented << "Forward complex FFT is not implemented yet";
}

P10Error Fft::forward_real(const Tensor& signal_in, Tensor& freq_out) const {
    if (signal_in.dims() != 2) {
        return P10Error::InvalidArgument << "Input tensor must have [N x T]";
    }
    P10_RETURN_IF_ERROR(check_input_device_and_dtype(signal_in));

    const auto type = signal_in.dtype();
    const auto num_samples = signal_in.shape(1).unwrap();
    const auto num_signals = signal_in.shape(0).unwrap();
    const auto num_ffts = num_samples / 2 + 1;

    double scalar_factor = 1.0;
    if (normalize_ == Normalize::ByN) {
        scalar_factor = 1.0 / static_cast<double>(num_samples);
    }

    freq_out.create(make_shape(num_signals, num_ffts, 2), type);

    pocketfft::shape_t shape_in = {size_t(num_signals), size_t(num_samples)};
    return type.match(
        [&](auto) -> P10Error {
            return P10Error::InvalidArgument << "Unsupported tensor dtype for FFT";
        },
        [&](auto t) -> P10Error {
            using scalar_t = decltype(t)::type;

            auto freq_out_s = freq_out.as_span2d<std::complex<scalar_t>>().unwrap();
            const auto signal_in_s = signal_in.as_span2d<const scalar_t>().unwrap();

            pocketfft::stride_t stride_in = {
                std::ptrdiff_t(num_samples * sizeof(scalar_t)),
                sizeof(scalar_t)
            };
            pocketfft::stride_t stride_out = {
                std::ptrdiff_t(num_ffts * sizeof(std::complex<scalar_t>)),
                sizeof(std::complex<scalar_t>)
            };

            pocketfft::r2c<scalar_t>(
                shape_in,
                stride_in,
                stride_out,
                1,
                pocketfft::FORWARD,
                signal_in_s.row(0),
                freq_out_s.row(0),
                scalar_t(scalar_factor)
            );

            return P10Error::Ok;
        }
    );
}

P10Error Fft::inverse(const Tensor&, Tensor&) const {
    return P10Error::NotImplemented << "Inverse complex FFT is not implemented yet";
}

P10Error Fft::inverse_real(const Tensor& freq_in, Tensor& signal_out) const {
    auto validation = validate_frequency_arguments(freq_in);
    if (validation.is_error()) {
        return validation.err();
    }
    const auto type = freq_in.dtype();
    const auto num_signals = validation.unwrap();
    const auto num_ffts = freq_in.shape(1).unwrap();
    const auto num_samples = (num_ffts - 1) * 2;

    P10_RETURN_IF_ERROR(signal_out.create(make_shape(num_signals, num_samples), type));

    double scalar_factor = 1.0;
    if (normalize_ == Normalize::ByN) {
        scalar_factor = 1.0 / static_cast<double>(num_samples);
    }

    pocketfft::shape_t shape_out = {size_t(num_signals), size_t(num_samples)};
    return type.match(
        [&](auto) -> P10Error {
            return P10Error::InvalidArgument << "Unsupported tensor dtype for FFT";
        },
        [&](auto t) -> P10Error {
            using scalar_t = decltype(t)::type;
            pocketfft::stride_t stride_in = {
                std::ptrdiff_t(num_ffts * sizeof(std::complex<scalar_t>)),
                sizeof(std::complex<scalar_t>)
            };

            pocketfft::stride_t stride_out = {
                std::ptrdiff_t(num_samples * sizeof(scalar_t)),
                sizeof(scalar_t)
            };

            const auto freq_in_s = freq_in.as_span2d<const std::complex<scalar_t>>().unwrap();
            auto signal_out_s = signal_out.as_span2d<scalar_t>().unwrap();

            pocketfft::c2r<scalar_t>(
                shape_out,
                stride_in,
                stride_out,
                1,
                pocketfft::BACKWARD,
                freq_in_s.row(0),
                signal_out_s.row(0),
                scalar_t(scalar_factor)
            );

            return P10Error::Ok;
        }
    );
}

namespace {
    P10Result<int64_t> validate_frequency_arguments(const Tensor& frequency) {
        if (frequency.dims() != 3) {
            return Err(
                P10Error::InvalidArgument << "Input tensor must have 3 dimensions [N x F x 2]"
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
}  // namespace
}  // namespace p10::op

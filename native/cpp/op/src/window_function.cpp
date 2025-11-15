#include "window_function.hpp"

#include <algorithm>
#include <cstddef>

#include <ptensor/tensor.hpp>

#include "elemwise.hpp"

namespace p10::op {
WindowFunction::WindowFunction(Function func) : func_(func) {}

P10Error WindowFunction::transform(const Tensor& input, Tensor& output) {
    if (input.dims() != 2) {
        return P10Error::InvalidArgument << "Input tensor must be 2D.";
    }
    P10_RETURN_IF_ERROR(output.create(input.shape(), input.dtype()));

    P10_RETURN_IF_ERROR(generate_window(input.shape(0).unwrap(), input.dtype()));

    return input.dtype().match([&](auto scalar) -> P10Error {
        using scalar_t = typename decltype(scalar)::type;

        auto input_span = input.as_span2d<scalar_t>().unwrap();
        auto output_span = output.as_span2d<scalar_t>().unwrap();
        auto window_span = window_->as_span1d<scalar_t>().unwrap();

        const auto num_signals = input_span.height();
        const auto num_samples = input_span.width();

        for (size_t signal_idx = 0; signal_idx < num_signals; ++signal_idx) {
            auto out_row = output_span.row(signal_idx);
            const auto in_row = input_span.row(signal_idx);
            std::transform(
                in_row,
                in_row + num_samples,
                window_span.begin(),
                out_row,
                std::multiplies<scalar_t>()
            );
        }
        return P10Error::Ok;
    });
}

P10Error
WindowFunction::transform_borders(const Tensor& input, Tensor& output, size_t border_size) {
    if (input.dims() != 2) {
        return P10Error::InvalidArgument << "Input tensor must be 2D.";
    }
    if (border_size * 2 > size_t(input.shape(1).unwrap())) {
        return P10Error::InvalidArgument << "Border size is too large for the input tensor.";
    }

    P10_RETURN_IF_ERROR(output.create(input.shape(), input.dtype()));

    P10_RETURN_IF_ERROR(generate_window(border_size * 2, input.dtype()));

    return input.dtype().match([&](auto scalar) -> P10Error {
        using scalar_t = typename decltype(scalar)::type;

        auto input_span = input.as_span2d<scalar_t>().unwrap();
        auto output_span = output.as_span2d<scalar_t>().unwrap();
        auto window_span = window_->as_span1d<scalar_t>().unwrap();

        const auto num_signals = input_span.height();
        const auto num_samples = input_span.width();

        for (size_t signal_idx = 0; signal_idx < num_signals; ++signal_idx) {
            auto out_row = output_span.row(signal_idx);
            const auto in_row = input_span.row(signal_idx);

            // Center part
            std::copy(
                in_row + border_size,
                in_row + num_samples - border_size,
                out_row + border_size
            );

            // Borders
            for (size_t n = 0; n < border_size; ++n) {
                out_row[n] = in_row[n] * window_span[n];
                out_row[num_samples - border_size + n] =
                    in_row[num_samples - border_size + n] * window_span[border_size + n];
            }
        }
        return P10Error::Ok;
    });
}

P10Error WindowFunction::generate_window(size_t size, Dtype type) {
    if (!window_) {
        window_ = std::make_unique<Tensor>();
    }

    P10_RETURN_IF_ERROR(window_->create(make_shape(size), type));

    window_->visit([&](auto out_span) {
        using scalar_t = typename decltype(out_span)::value_type;

        for (size_t n = 0; n < size; ++n) {
            scalar_t value = 1.0;
            switch (func_) {
                case Function::Hanning:
                    value = static_cast<scalar_t>(
                        0.5 * (1.0 - std::cos((2.0 * M_PI * n) / (size - 1)))
                    );
                    break;
                case Function::Hamming:
                    value = static_cast<scalar_t>(
                        0.54 - 0.46 * std::cos((2.0 * M_PI * n) / (size - 1))
                    );
                    break;
                case Function::Identity:
                    value = static_cast<scalar_t>(1.0);
                    break;
            }
            out_span[n] = value;
        }
    });
    return P10Error::Ok;
}

}  // namespace p10::op
#include "blur.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

#include <immintrin.h>  // AVX2 intrinsics
#include <ptensor/dtype.hpp>
#include <ptensor/p10_error.hpp>
#include <ptensor/tensor.hpp>

#include "fixed_point_type.hpp"

namespace p10::op {

namespace {
    void create_gaussian_kernel_(std::span<float> kernel, float sigma);

    template<typename scalar_t, typename fixed_t>
    void apply_horizontal_kernel(
        Accessor2D<const scalar_t> input,
        Accessor2D<scalar_t> output,
        std::span<const fixed_t> kernel,
        fixed_t factor
    ) {
        const int half_size = int(kernel.size()) / 2;
        const int height = int(input.rows());
        const int width = int(input.cols());

        for (int y = 0; y < height; ++y) {
            const auto input_row = input[y];
            auto output_row = output[y];
            for (int x = 0; x < width; ++x) {
                fixed_t sum = 0;
                for (int k = -half_size; k <= half_size; ++k) {
                    const int xx = std::clamp(x + k, 0, width - 1);
                    sum += fixed_t(input_row[xx]) * kernel[k + half_size];
                }
                output_row[x] = scalar_t(sum / factor);
            }
        }
    }

    template<typename scalar_t, typename fixed_t>
    void apply_vertical_kernel(
        Accessor2D<const scalar_t> input,
        Accessor2D<scalar_t> output,
        std::span<const fixed_t> kernel,
        fixed_t factor
    ) {
        const int half_size = int(kernel.size()) / 2;
        const int height = int(input.rows());
        const int width = int(input.cols());
        for (int y = 0; y < height; ++y) {
            auto output_row = output[y];
            for (int x = 0; x < width; ++x) {
                fixed_t sum = 0;
                for (int k = -half_size; k <= half_size; ++k) {
                    const int yy = std::clamp(y + k, 0, height - 1);
                    sum += fixed_t(input[yy][x]) * kernel[k + half_size];
                }
                output_row[x] = scalar_t(sum / factor);
            }
        }
    }

}  // namespace

P10Result<GaussianBlur> GaussianBlur::create(size_t kernel_size, float sigma) {
    if (kernel_size % 2 == 0 || kernel_size > MAX_KERNEL_SIZE) {
        return Err(
            P10Error::InvalidArgument,
            "Kernel size must be an odd number and less than or equal to "
            "MAX_KERNEL_SIZE."
        );
    }
    KernelStorage kernel_stack_data;
    std::span<float> kernel {kernel_stack_data.data(), kernel_size};
    create_gaussian_kernel_(kernel, sigma);

    return Ok(GaussianBlur(kernel_stack_data, kernel_size));
}

namespace {
    void create_gaussian_kernel_(std::span<float> kernel, float sigma) {
        float sum = 0.0f;
        int half_size = int(kernel.size() / 2);
        for (int i = -half_size; i <= half_size; ++i) {
            kernel[i + half_size] = static_cast<float>(
                std::exp(-(i * i) / (2 * sigma * sigma))
                / (sigma * std::sqrt(2.0f * std::numbers::pi))
            );
            sum += kernel[i + half_size];
        }

        for (size_t i = 0; i < kernel.size(); ++i) {
            kernel[i] /= sum;
        }
    }

}  // namespace

P10Error GaussianBlur::transform(const Tensor& input, Tensor& output) {
    if (input.shape().dims() != 3) {
        return P10Error::InvalidArgument << "Input tensor must be a 3D image with shape (H, W, C).";
    }

    const Dtype dtype = input.dtype();

    if (horizontal_out_ == nullptr) {
        horizontal_out_ = std::make_shared<Tensor>();
    }
    P10_RETURN_IF_ERROR(horizontal_out_->create(input.shape(), dtype));
    P10_RETURN_IF_ERROR(output.create(input.shape(), dtype));

    return dtype.match([&](auto type_tag) -> P10Error {
        using scalar_t = typename decltype(type_tag)::type;

        if constexpr (detail::has_fixed_point_type<scalar_t>::value) {
            auto input_acc = input.as_accessor3d<const scalar_t>().unwrap();
            auto horizontal_out_acc = horizontal_out_->as_accessor3d<scalar_t>().unwrap();
            auto output_acc = output.as_accessor3d<scalar_t>().unwrap();

            const auto float_kernel = get_kernel();
            using fixed_t = typename detail::fixed_point_type<scalar_t>::type;
            std::array<fixed_t, MAX_KERNEL_SIZE> kernel;
            const fixed_t factor = detail::fixed_point_type<scalar_t>::factor;
            std::transform(
                float_kernel.begin(),
                float_kernel.end(),
                kernel.begin(),
                [=](float value) { return fixed_t(value * factor); }
            );
            std::span<const fixed_t> kernel_span {kernel.data(), get_kernel().size()};

            for (int64_t channel_plane = 0; channel_plane < input_acc.channels(); channel_plane++) {
                apply_horizontal_kernel<scalar_t, fixed_t>(
                    input_acc[channel_plane],
                    horizontal_out_acc[channel_plane],
                    kernel_span,
                    factor
                );
                apply_vertical_kernel<scalar_t, fixed_t>(
                    horizontal_out_acc[channel_plane].as_const(),
                    output_acc[channel_plane],
                    kernel_span,
                    factor
                );
            }
        } else {
            return P10Error::InvalidArgument << "Unsupported data type for this operation.";
        }
        return P10Error::Ok;
    });
}

}  // namespace p10::op

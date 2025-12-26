#include "blur.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

#include <immintrin.h>  // AVX2 intrinsics
#include <ptensor/simd/bitwise_math.hpp>
#include <ptensor/dtype.hpp>
#include <ptensor/p10_error.hpp>
#include <ptensor/tensor.hpp>

#include <type_traits>

#include "accumulator_traits.hpp"

namespace p10::op {
using detail::accumulator_traits;

namespace {
    void create_gaussian_kernel_(std::span<float> kernel, float sigma);

#if 0
    template<typename scalar_t, typename accum_t>
    void apply_1d_kernel_generic(
        std::span<const scalar_t> input,
        std::span<scalar_t> output,
        std::span<const accum_t> kernel,
        int kernel_half_size
    ) {
        for (int x = 0; x < int(input.size()); ++x) {
            accumulator_traits<scalar_t> sum = 0;
            for (int k = -kernel_half_size; k <= kernel_half_size; ++k) {
                const int xx = x + k;
                if (xx < 0 || xx >= int(input.size())) {
                    continue;
                }
                sum += accum_t(input[xx]) * kernel[k + kernel_half_size];
            }
            output[x] = accumulator_traits<scalar_t>::to_scalar(sum);
        }
    }

    template<typename scalar_t, typename accum_t>
    void apply_1d_kernel_8_generic(
        scalar_t* input,
        const scalar_t* output,
        const accum_t* kernel,
        int kernel_half_size
    ) {
        for (int x = 0; x < 8; ++x) {
            accumulator_traits<scalar_t> sum = 0;
            for (int k = -kernel_half_size; k <= kernel_half_size; ++k) {
                sum += accum_t(input[x + k]) * kernel[k + kernel_half_size];
            }
            output[x] = accumulator_traits<scalar_t>::to_scalar(sum);
        }
    }

    __attribute__((target("avx2"))) void apply_1d_kernel_8_avx(
        float* input,
        const float* output,
        const float* kernel,
        int kernel_half_size
    ) {
        // TODO: implement AVX version
    }
#endif

    template<typename scalar_t, typename accum_t>
    void apply_1d_kernel(
        Span2D<const scalar_t> input,
        Span2D<scalar_t> output,
        std::span<const accum_t> kernel
    ) {
        constexpr size_t BLOCK_SIZE = 64;
        constexpr size_t SIMD_SIZE = 8;

        const auto rows = input.rows();
        const auto cols = input.cols();
        const bool avx2_supported = false;  //is_avx2_supported();

        const auto kernel_half_size = int(kernel.size() / 2);
        const auto aligned_max_cols = cols - bitwise_modulo<BLOCK_SIZE>(cols) - kernel.size() + 1;

        for (size_t row = 0; row < rows; row++) {
            apply_1d_kernel_generic(
                input.row_span(row).subspan(0, kernel_half_size),
                output.row_span(row).subspan(0, kernel_half_size),
                kernel,
                kernel_half_size
            );
            apply_1d_kernel_generic(
                input.row_span(row).subspan(cols - kernel_half_size, kernel_half_size),
                output.row_span(row).subspan(cols - kernel_half_size, kernel_half_size),
                kernel,
                kernel_half_size
            );

            const auto input_row = input.row(row);
            auto output_row = output.row(row);

            for (size_t block_col = kernel_half_size; block_col < aligned_max_cols;
                 block_col += BLOCK_SIZE) {
                for (size_t simd_col = block_col; simd_col < block_col + BLOCK_SIZE;
                     simd_col += SIMD_SIZE) {
                    const scalar_t* input_block = &input_row[simd_col];
                    scalar_t* output_block = &output.row(row)[simd_col];

                    if constexpr (std::is_same_v<scalar_t, float>) {
                        if (avx2_supported) {
                            apply_1d_kernel_8_avx(
                                input_block,
                                output_block,
                                kernel.data(),
                                kernel_half_size
                            );
                            continue;
                        }
                    }
                    apply_1d_kernel_8_generic(
                        input_block,
                        output_block,
                        reinterpret_cast<const accum_t*>(kernel.data()),
                        kernel_half_size
                    );
                }
            }
        }
    }

    template<typename scalar_t, typename fixed_t>
    void apply_horizontal_kernel(
        Accessor2D<const scalar_t> input,
        Accessor2D<scalar_t> output,
        std::span<const fixed_t> kernel
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
                output_row[x] = scalar_t(accumulator_traits<scalar_t>::to_scalar(sum));
            }
        }
    }

    template<typename scalar_t, typename fixed_t>
    void apply_vertical_kernel(
        Accessor2D<const scalar_t> input,
        Accessor2D<scalar_t> output,
        std::span<const fixed_t> kernel
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
                output_row[x] = scalar_t(accumulator_traits<scalar_t>::to_scalar(sum));
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
        using accum_t = accumulator_traits<scalar_t>::accum_type;

        if constexpr (!std::is_same_v<accum_t, detail::AccumulatorNotDefined>) {
            auto input_acc = input.as_accessor3d<const scalar_t>().unwrap();
            auto horizontal_out_acc = horizontal_out_->as_accessor3d<scalar_t>().unwrap();
            auto output_acc = output.as_accessor3d<scalar_t>().unwrap();

            const auto float_kernel = get_kernel();

            std::array<accum_t, MAX_KERNEL_SIZE> kernel;
            std::transform(
                float_kernel.begin(),
                float_kernel.end(),
                kernel.begin(),
                [](float value) { return accumulator_traits<scalar_t>::from_float(value); }
            );

            std::span<const accum_t> kernel_span {kernel.data(), get_kernel().size()};

            for (int64_t channel_plane = 0; channel_plane < input_acc.channels(); channel_plane++) {
                apply_horizontal_kernel<scalar_t, accum_t>(
                    input_acc[channel_plane],
                    horizontal_out_acc[channel_plane],
                    kernel_span
                );
                apply_vertical_kernel<scalar_t, accum_t>(
                    horizontal_out_acc[channel_plane].as_const(),
                    output_acc[channel_plane],
                    kernel_span
                );
            }
            return P10Error::Ok;
        } else {
            return P10Error::InvalidArgument << "Unsupported data type for this operation.";
        }
        
    });
}

}  // namespace p10::op

#include "blur.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>

#include <ptensor/dtype.hpp>
#include <ptensor/p10_error.hpp>
#include <ptensor/tensor.hpp>

#include <type_traits>

#include "accumulator_traits.hpp"
#include "p10_internal/simd/tile2d.hpp"
#include "ptensor/tile_region2d.hpp"

#include "blur.fast.hpp"

namespace p10::op {
using detail::accumulator_traits;

namespace {
    void create_gaussian_kernel(std::span<float> kernel, float sigma);

    template<typename scalar_t, size_t MAX_SIZE>
    auto convert_array(std::span<const float> in) {
        using fixed_t = typename accumulator_traits<scalar_t>::accum_type;
        std::array<fixed_t, MAX_SIZE> out;
        std::transform(in.begin(), in.end(), out.begin(), [](float value) {
            return accumulator_traits<scalar_t>::from_float(value);
        });
        return out;
    }

    // ---- Generic path: any dtype with an accumulator, 3D (H, W, C) image -----

    template<typename scalar_t, typename fixed_t>
    void blur_kernel(
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

}  // namespace

P10Result<GaussianBlur> GaussianBlur::create(size_t kernel_size, float sigma) {
    if (kernel_size % 2 == 0 || kernel_size > MAX_KERNEL_SIZE) {
        return Err(
            P10Error::InvalidArgument,
            "Kernel size must be an odd number and less than or equal to "
            "MAX_KERNEL_SIZE."
        );
    }
    GaussianBlur new_blur {kernel_size};
    create_gaussian_kernel(new_blur.kernel_.get(), sigma);
    return Ok(std::move(new_blur));
}

namespace {
    void create_gaussian_kernel(std::span<float> kernel, float sigma) {
        float sum = 0.0F;
        int const half_size = static_cast<int>(kernel.size() / 2);
        for (int i = -half_size; i <= half_size; ++i) {
            kernel[i + half_size] = static_cast<float>(
                std::exp(-(i * i) / (2 * sigma * sigma))
                / (sigma * std::sqrt(2.0F * std::numbers::pi))
            );
            sum += kernel[i + half_size];
        }

        for (auto& value : kernel) {
            value /= sum;
        }
    }
}  // namespace

P10Error GaussianBlur::transform(const Tensor& input, Tensor& output) {
    const Dtype dtype = input.dtype();

    if (input.shape().dims() != 3) {
        return P10Error::InvalidArgument << "Input must be a 3D image with shape (H, W, C).";
    }

    P10_RETURN_IF_ERROR(horizontal_out_.create(input.shape(), dtype));
    P10_RETURN_IF_ERROR(output.create(input.shape(), dtype));

    return dtype.match([&](auto type_tag) -> P10Error {
        using scalar_t = decltype(type_tag)::type;
        using accum_t = accumulator_traits<scalar_t>::accum_type;

        if constexpr (!std::is_same_v<accum_t, detail::AccumulatorNotDefined>) {
            auto input_acc = input.as_planar_span3d<const scalar_t>().unwrap();
            auto horizontal_out_acc = horizontal_out_.as_planar_span3d<scalar_t>().unwrap();
            auto output_acc = output.as_planar_span3d<scalar_t>().unwrap();

            const auto height = input_acc.height();
            const auto width = input_acc.width();

            auto kernel = convert_array<scalar_t, MAX_KERNEL_SIZE>(kernel_.get());
            const std::span<const accum_t> kernel_span {kernel.data(), kernel_.size};
            
            const auto first_pass_kernel = [&](const TileRegion2D& region) {
                for (int64_t channel_plane = 0; channel_plane < input_acc.channels();
                     channel_plane++) {
                    Accessor2D<const scalar_t> src = input_acc[channel_plane](region).as_const();
                    blur_kernel(
                        src,
                        horizontal_out_acc[channel_plane](region),
                        kernel_span
                    );
                }
            };

            const auto second_pass_kernel = [&](const TileRegion2D& region) {
                for (int64_t channel_plane = 0; channel_plane < input_acc.channels();
                     channel_plane++) {
                    Accessor2D<const scalar_t> src = horizontal_out_acc[channel_plane](region).as_const();
                    blur_kernel(
                        src,
                        output_acc[channel_plane](region),
                        kernel_span
                    );
                }
            };

            if (!fastblur::try_fast_blur<scalar_t>(input_acc, horizontal_out_acc, kernel_.get())) {
                simd::tile2d<scalar_t>(height, width, first_pass_kernel, simd::Portable<32>(first_pass_kernel));
            }
            horizontal_out_.transpose();
            auto horizontal_in_acc = horizontal_out_.as_planar_span3d<const scalar_t>().unwrap();
            if (!fastblur::try_fast_blur<scalar_t>(horizontal_in_acc, output_acc, kernel_.get())) {
                simd::tile2d<scalar_t>(height, width, second_pass_kernel, simd::Portable<32>(second_pass_kernel));                
            }

            output.transpose();
                
            return P10Error::Ok;
        } else {
            return P10Error::InvalidArgument << "Unsupported data type for this operation.";
        }
    });
}

}  // namespace p10::op

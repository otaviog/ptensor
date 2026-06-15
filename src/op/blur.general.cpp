#include "blur.hpp"

#include <cmath>
#include <numbers>

#include <ptensor/dtype.hpp>
#include <ptensor/p10_error.hpp>
#include <ptensor/tensor.hpp>

#include <type_traits>

#include "p10_internal/simd/tile2d.hpp"
#include "ptensor/region2d.hpp"

#include "blur.fast.hpp"

namespace p10::op {

namespace {
    void create_gaussian_kernel(std::span<float> kernel, float sigma);

    // Generic horizontal blur for any arithmetic dtype (scalar fallback path,
    // runtime kernel size). Shares the tap loop with the float fast path via
    // hblur_row. The given accessor is the whole region to blur, so clamp the
    // apron to its own extent.
    template<typename scalar_t>
    void blur_kernel(
        Accessor2D<const scalar_t> input,
        Accessor2D<scalar_t> output,
        std::span<const float> kernel
    ) {
        const int half = int(kernel.size()) / 2;
        const int64_t width = input.cols();
        for (int64_t row = 0; row < input.rows(); ++row) {
            hblur_row<scalar_t>(
                input[row], output[row], 0, width, half, kernel.data(), width, /*clamp_edges=*/true
            );
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

    // Accept a 2D plane (H, W) or a planar 3D image (C, H, W). RankFit::Flexible
    // promotes the 2D case to a single plane.
    if (input.shape().dims() != 2 && input.shape().dims() != 3) {
        return P10Error::InvalidArgument << "Input must be a 2D plane or a planar 3D image.";
    }

    P10_RETURN_IF_ERROR(scratch_.create(input.shape(), dtype));

    return dtype.match([&](auto type_tag) -> P10Error {
        using scalar_t = decltype(type_tag)::type;

        if constexpr (std::is_arithmetic_v<scalar_t>) {
            const auto kernel = kernel_.get();

            // Separable blur with a transpose between passes: a horizontal blur,
            // transpose, a second horizontal blur (= vertical), transpose back.
            {
                auto src = input.as_span3d<const scalar_t, RankFit::Flexible>().unwrap();
                auto dst = scratch_.as_span3d<scalar_t, RankFit::Flexible>().unwrap();
                const auto pass = [&](const Region2D& region) {
                    for (int64_t plane = 0; plane < src.channels(); plane++) {
                        blur_kernel(src[plane](region).as_const(), dst[plane](region), kernel);
                    }
                };
                if (!fastblur::try_fast_blur<scalar_t>(src, dst, kernel)) {
                    simd::tile2d<scalar_t>(src.rows(), src.cols(), pass, simd::Portable<32>(pass));
                }
            }

            scratch_.transpose();
            P10_RETURN_IF_ERROR(output.create(scratch_.shape(), dtype));

            {
                auto src = scratch_.as_span3d<const scalar_t, RankFit::Flexible>().unwrap();
                auto dst = output.as_span3d<scalar_t, RankFit::Flexible>().unwrap();
                const auto pass = [&](const Region2D& region) {
                    for (int64_t plane = 0; plane < src.channels(); plane++) {
                        blur_kernel(src[plane](region).as_const(), dst[plane](region), kernel);
                    }
                };
                if (!fastblur::try_fast_blur<scalar_t>(src, dst, kernel)) {
                    simd::tile2d<scalar_t>(src.rows(), src.cols(), pass, simd::Portable<32>(pass));
                }
            }

            output.transpose();
            return P10Error::Ok;
        } else {
            return P10Error::InvalidArgument << "Unsupported data type for this operation.";
        }
    });
}

}  // namespace p10::op

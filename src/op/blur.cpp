#include "blur.hpp"

#include <cmath>
#include <numbers>

#include <ptensor/dtype.hpp>
#include <ptensor/p10_error.hpp>
#include <ptensor/tensor.hpp>

#include <type_traits>

#include "blur.fast.hpp"
#include "p10_internal/simd/tile2d.hpp"
#include "ptensor/region2d.hpp"

namespace p10::op {

template<typename T>
concept Scalar = std::is_arithmetic_v<T>;

namespace {
    void create_gaussian_kernel(std::span<float> kernel, float sigma);

    // Generic horizontal blur for any arithmetic dtype (scalar fallback path,
    // runtime kernel size). Shares the tap loop with the float fast path via
    // hblur_row. The given accessor is the whole region to blur, so clamp the
    // apron to its own extent.
    template<Scalar scalar_t>
    void blur_kernel(
        Accessor2D<const scalar_t> input,
        Accessor2D<scalar_t> output,
        std::span<const float> kernel
    ) {
        const int half = int(kernel.size()) / 2;
        const int64_t width = input.cols();
        for (int64_t row = 0; row < input.rows(); ++row) {
            hblur_row<scalar_t>(
                input[row],
                output[row],
                0,
                width,
                half,
                kernel.data(),
                width,
                /*clamp_edges=*/true
            );
        }
    }

    template<Scalar scalar_t, size_t KHALF>
    void hblur_pass_with_border(
        Span3D<const scalar_t> src,
        Span3D<scalar_t> dst,
        std::span<const float_t> kernel
    ) {
        auto channel_wise_pass = [&](auto blur_func, const auto in, auto out) {
            return [&](const Region2D region) {
                for (int64_t channel = 0; channel < in.channels(); ++channel) {
                    const auto in_region = in[channel](region);
                    auto out_region = out[channel](region);
                    for (int64_t row = 0; row < in.rows(); ++row) {
                        blur_func(in_region[row], out_region[row], kernel);
                    }
                }
            };
        };

        constexpr simd::TileBorder BORDER {.horizontal = KHALF, .vertical = 0};
        const Region2D full {.row = 0, .col = 0, .height = src.rows(), .width = src.cols()};

        simd::tile2d<float, simd::TileExecution::SEQUENTIAL, BORDER>(
            src.rows(),
            dst.cols(),
            channel_wise_pass(hblur_scalar),

                make_hblur_border<KHALF>(in, out, kernel),
            make_avx2_hblur<KHALF>(in, out, kernel),
            make_neon_hblur<KHALF>(in, out, kernel),
            make_portable_hblur<KHALF>(in, out, kernel)
        );
    }

    template<Scalar scalar_t>
    void
    hblur_pass(Span3D<const scalar_t> src, Span3D<scalar_t> dst, std::span<const float_t> kernel) {
        const size_t khalf = kernel.size() >> 1;

        switch (khalf) {
            case 1:
                hblur_pass_with_border<1>(src, dst, kernel);
                break;
            case 2:
                hblur_pass_with_border<2>(src, dst, kernel);
                break;
            case 3:
                hblur_pass_with_border<3>(src, dst, kernel);
                break;
            case 4:
                hblur_pass_with_border<4>(src, dst, kernel);
                break;
            default:
                hblur_pass_with_border<0>(src, dst, kernel);
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

P10Error GaussianBlur::transform(const Tensor& input, Tensor& output) {
    const Dtype dtype = input.dtype();

    if (input.shape().dims() < 2) {
        return P10Error::InvalidArgument << "Input tensor must have at least 2 dimensions.";
    }

    P10_RETURN_IF_ERROR(scratch_.create(input.shape(), dtype));
    P10_RETURN_IF_ERROR(output.create(input.shape(), dtype));

    return dtype.match([&](auto type_tag) -> P10Error {
        using scalar_t = decltype(type_tag)::type;

        if constexpr (std::is_arithmetic_v<scalar_t>) {
            const auto kernel = kernel_.get();

            const auto in = input.as_span3d<const scalar_t, RankFit::Flexible>().unwrap();
            auto scratch = scratch_.as_span3d<scalar_t, RankFit::Flexible>().unwrap();

            hblur_pass<scalar_t>(in, scratch, kernel);
            scratch_.transpose();
            auto out = output.as_span3d<scalar_t, RankFit::Flexible>().unwrap();
            hblur_pass<scalar_t>(scratch.as_const(), out, kernel);
            output.transpose();
            return P10Error::Ok;
        } else {
            return P10Error::InvalidArgument << "Unsupported data type for this operation.";
        }
    });
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

}  // namespace p10::op

#include "blur.hpp"

#include <cmath>
#include <numbers>

#include <ptensor/dtype.hpp>
#include <ptensor/p10_error.hpp>
#include <ptensor/tensor.hpp>

#include <type_traits>

#include "blur.hblur.portable.hpp"
#include "p10_internal/simd/tile2d.hpp"
#include "ptensor/region2d.hpp"

namespace p10::op {

template<typename T>
concept Scalar = std::is_arithmetic_v<T>;

namespace {
    void create_gaussian_kernel(std::span<float> kernel, float sigma);

    template<Scalar scalar_t, size_t KHALF>
    void hblur_pass_impl(
        Span3D<const scalar_t> src,
        Span3D<scalar_t> dst,
        std::span<const float> kernel,
        auto scalar_kernel
    ) {
        const int half = static_cast<int>(kernel.size()) / 2;
        const simd::TileBorder border {.horizontal = half, .vertical = 0};
        const Region2D full {.row = 0, .col = 0, .height = src.rows(), .width = src.cols()};

        const auto simd_kernel = [=](const Region2D& region) {
            const int64_t col_end = region.col + region.width;
            const int64_t row_end = region.row + region.height;
            for (int64_t channel = 0; channel < src.channels(); ++channel) {
                for (int64_t row = region.row; row < row_end; ++row) {
                    hblur_portable<scalar_t, KHALF>(
                        src[channel](row).as_const(),
                        dst[channel].as_accessor().transpose()[row],
                        region.col,
                        col_end,
                        kernel.data()
                    );
                }
            };

            simd::tile2d<scalar_t, simd::TileExecution::SEQUENTIAL>(
                src.rows(),
                dst.cols(),
                border,
                scalar_kernel,
                simd::Portable<8, scalar_t>(simd_kernel)
            );
        }
    }

    // One horizontal blur pass over every channel plane, runtime kernel size.
    // tile2d carries the +/-half apron as a runtime border: the interior runs
    // the SIMD (or portable) kernel without edge checks while the scalar sweep
    // clamps the frame. Only float has SIMD kernels; tile2d's dtype dispatch
    // drops them for other types, which fall through to the scalar sweep below.
    template<Scalar scalar_t>
    void
    hblur_pass(Span3D<const scalar_t> src, Span3D<scalar_t> dst, std::span<const float> kernel) {
        const int half = static_cast<int>(kernel.size()) / 2;
        const simd::TileBorder border {.horizontal = half, .vertical = 0};
        const Region2D full {.row = 0, .col = 0, .height = src.rows(), .width = src.cols()};

        // Clamping scalar sweep: serves the edge frame, and the whole plane
        // when scalar_t has no SIMD kernel.
        const auto scalar_edge = [=](const Region2D& region) {
            const int64_t col_end = region.col + region.width;
            const int64_t row_end = region.row + region.height;
            for (int64_t channel = 0; channel < src.channels(); ++channel) {
                for (int64_t row = region.row; row < row_end; ++row) {
                    hblur_scalar<scalar_t>(
                        src[channel][row],
                        dst[channel].as_accessor().transpose()[row],
                        region.col,
                        col_end,
                        half,
                        kernel.data(),
                        src.cols(),
                        /*clamp_edges=*/true
                    );
                }
            }
        };

        switch (half) {
            case 1:
                hblur_pass_impl<scalar_t, 1>(src, dst, kernel, scalar_edge);
                break;
            case 2:
                hblur_pass_impl<scalar_t, 2>(src, dst, kernel, scalar_edge);
                break;
            case 3:
                hblur_pass_impl<scalar_t, 3>(src, dst, kernel, scalar_edge);
                break;
            default:
                simd::tile2d<scalar_t>(src.rows(), src.cols(), simd::TileBorder{}, scalar_edge);
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

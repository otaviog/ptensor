#include "blur.hpp"

#include <array>
#include <cmath>
#include <numbers>

#include <ptensor/dtype.hpp>
#include <ptensor/p10_error.hpp>
#include <ptensor/tensor.hpp>

#include <type_traits>

#include "blur.hblur.avx2.hpp"
#include "blur.hblur.hpp"
#include "blur.hblur.neon.hpp"
#include "p10_internal/simd/tile2d.hpp"
#include "ptensor/region2d.hpp"

namespace p10::op {

template<typename T>
concept Scalar = std::is_arithmetic_v<T>;

namespace {
    void create_gaussian_kernel(std::span<float> kernel, float sigma);

    // Input shape with its last two (plane) dims swapped: the layout a single
    // transposing hblur_pass produces.
    Shape plane_transposed(const Shape& shape) {
        const size_t dims = shape.dims();
        std::array<size_t, P10_MAX_SHAPE> perm {};
        for (size_t i = 0; i < dims; ++i) {
            perm[i] = i;
        }
        std::swap(perm[dims - 1], perm[dims - 2]);
        return shape.permute(std::span<const size_t>(perm.data(), dims)).unwrap();
    }

    // Interior pass for a fixed half-width KHALF: the tap loop unrolls because
    // KHALF is a compile-time constant. tile2d runs the first kernel the target
    // supports on the halo-inset interior and hands the clamped frame to
    // scalar_kernel. The AVX2/NEON specs are tagged float, so tile2d drops them
    // for other dtypes, which fall through to the portable interior.
    template<Scalar scalar_t, size_t KHALF>
    void hblur_pass_impl(
        Span3D<const scalar_t> src,
        Span3D<scalar_t> dst,
        std::span<const float> kernel,
        auto scalar_kernel
    ) {
        const simd::TileBorder border {.horizontal = KHALF, .vertical = 0};

        simd::tile2d<scalar_t, simd::TileExecution::PARALLEL>(
            src.rows(),
            src.cols(),
            border,
            scalar_kernel,
            make_avx2_hblur<scalar_t, KHALF>(src, dst, kernel.data()),
            make_neon_hblur<scalar_t, KHALF>(src, dst, kernel.data()),
            make_portable_hblur<scalar_t, KHALF>(src, dst, kernel.data())
        );
    }

    // One horizontal blur pass over every channel plane, runtime kernel size.
    // tile2d carries the +/-half apron as a runtime border: the interior runs
    // the unrolled kernel without edge checks while the scalar sweep clamps the
    // frame. Half-widths past the unrolled cases fall back to a scalar-only
    // tiling.
    template<Scalar scalar_t>
    void
    hblur_pass(Span3D<const scalar_t> src, Span3D<scalar_t> dst, std::span<const float> kernel) {
        const int half = static_cast<int>(kernel.size()) / 2;
        const auto scalar_edge = make_scalar_hblur<scalar_t>(src, dst, kernel.data(), half);

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
                simd::tile2d<scalar_t>(src.rows(), src.cols(), simd::TileBorder {}, scalar_edge);
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

    // Each hblur_pass writes its result transposed, so the intermediate holds a
    // plane-transposed image; the second pass transposes it back to the input
    // layout. Size scratch with the plane dims swapped accordingly.
    P10_RETURN_IF_ERROR(scratch_.create(plane_transposed(input.shape()), dtype));
    P10_RETURN_IF_ERROR(output.create(input.shape(), dtype));

    return dtype.match([&](auto type_tag) -> P10Error {
        using scalar_t = decltype(type_tag)::type;

        if constexpr (std::is_arithmetic_v<scalar_t>) {
            const auto kernel = kernel_.get();

            const auto in = input.as_span3d<const scalar_t, RankFit::Flexible>().unwrap();
            auto scratch = scratch_.as_span3d<scalar_t, RankFit::Flexible>().unwrap();
            auto out = output.as_span3d<scalar_t, RankFit::Flexible>().unwrap();

            hblur_pass<scalar_t>(in, scratch, kernel);
            hblur_pass<scalar_t>(scratch.as_const(), out, kernel);
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

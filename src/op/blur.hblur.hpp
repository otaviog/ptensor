#include <cstdint>
#include <memory>
#include <span>

#include <p10_internal/simd/tile2d.hpp>
#include <ptensor/dtype.hpp>
#include <ptensor/span2d.hpp>
#include <ptensor/tensor.hpp>

#include "blur.hblur.avx2.hpp"
#include "blur.hblur.neon.hpp"
#include "blur.hblur.portable.hpp"
#include "ptensor/accessor2d.hpp"
#include "ptensor/span3d.hpp"

namespace p10::op::fastblur {

namespace {
    template<int KHALF>
    void hblur_pass(Span3D<const float> input, Span3D<float> output, const float* kernel) {
        constexpr simd::TileBorder BORDER {.horizontal = KHALF, .vertical = 0};
        const Region2D full {
            .row = 0, .col = 0, .height = input.rows(), .width = input.cols()
        };
        for (int64_t channel_plane = 0; channel_plane < input.channels(); channel_plane++) {
            Accessor2D<const float> in = input[channel_plane](full);
            Accessor2D<float> out = output[channel_plane](full);
            simd::tile2d<float, simd::TileExecution::SEQUENTIAL, BORDER>(
                input.rows(),
                input.cols(),
                make_hblur_border<KHALF>(in, out, kernel),
                make_avx2_hblur<KHALF>(in, out, kernel),
                make_neon_hblur<KHALF>(in, out, kernel),
                make_portable_hblur<KHALF>(in, out, kernel)
            );
        }
    }
}  // namespace

template<typename scalar_t>
constexpr bool try_fast_blur(
    Span3D<const scalar_t> src,
    Span3D<scalar_t> dst,
    std::span<const float> kernel) {
    (void) src;
    (void) dst;
    (void) kernel;
    return false;
}

template<>
inline bool try_fast_blur<float>(
    Span3D<const float> src,
    Span3D<float> dst,
    std::span<const float> kernel
) {
    size_t khalf = kernel.size() >> 1;
    switch (khalf) {
        case 1:
            hblur_pass<1>(src, dst, kernel.data());
            break;
        case 2:
            hblur_pass<2>(src, dst, kernel.data());
            break;
        case 3:
            hblur_pass<3>(src, dst, kernel.data());
            break;
        case 4:
            hblur_pass<4>(src, dst, kernel.data());
            break;
        default:
            return false;
    }
    return true;
}

}  // namespace p10::op::fastblur

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <ptensor/span2d.hpp>
#include <p10_internal/simd/tile2d.hpp>

namespace p10::op {

// Accumulate in float for small integer types, double for >=32-bit ints; floats
// pass through. The scalar/portable convolution shares this with the generic
// fallback in blur.general.cpp.
template<typename scalar_t>
using blur_accum_t = std::conditional_t<
    std::is_floating_point_v<scalar_t>,
    scalar_t,
    std::conditional_t<(sizeof(scalar_t) >= 4), double, float>>;

// Stores an accumulator into the output scalar: round + saturate for integers,
// passthrough for floats (so the float fast path is bit-identical).
template<typename scalar_t>
inline scalar_t blur_store(blur_accum_t<scalar_t> value) {
    if constexpr (std::is_floating_point_v<scalar_t>) {
        return static_cast<scalar_t>(value);
    } else {
        using limits = std::numeric_limits<scalar_t>;
        using accum_t = blur_accum_t<scalar_t>;
        return static_cast<scalar_t>(std::clamp(
            std::round(value),
            static_cast<accum_t>(limits::min()),
            static_cast<accum_t>(limits::max())
        ));
    }
}

// One horizontal convolution sweep over columns [col_begin, col_end) of a row.
// `half` is the kernel radius; pass a compile-time constant to let the tap loop
// unroll. clamp_edges clamps the apron to [0, width); interior SIMD tiles skip
// it because the tiler guarantees the +/-half apron is in bounds.
template<typename scalar_t>
inline void hblur_row(
    Accessor1D<const scalar_t> in_row,
    Accessor1D<scalar_t> out_row,
    int64_t col_begin,
    int64_t col_end,
    int half,
    const float* kernel,
    int64_t width,
    bool clamp_edges
) {
    using accum_t = blur_accum_t<scalar_t>;
    for (int64_t col = col_begin; col < col_end; ++col) {
        accum_t sum = 0;
        for (int k = -half; k <= half; ++k) {
            int64_t src_col = col + k;
            if (clamp_edges) {
                src_col = std::clamp<int64_t>(src_col, 0, width - 1);
            }
            sum += static_cast<accum_t>(in_row[src_col]) * static_cast<accum_t>(kernel[k + half]);
        }
        out_row[col] = blur_store<scalar_t>(sum);
    }
}

// Horizontal 1D convolution over a tile region. KHALF is the kernel half-size
// (radius), known at compile time so the tap loop unrolls. The interior SIMD
// tiles run with clamp_edges == false (the tiler guarantees the +/-KHALF apron
// is in bounds); the edge frame runs the same loop with clamping.
template<int KHALF>
inline void hblur_region(
    Accessor2D<const float> input,
    Accessor2D<float> output,
    const float* kernel,
    const Region2D& region,
    bool clamp_edges
) {
    const int64_t width = input.cols();
    const int64_t row_end = region.row + region.height;
    const int64_t col_end = region.col + region.width;

    for (int64_t row = region.row; row < row_end; ++row) {
        hblur_row<float>(
            input[row], output[row], region.col, col_end, KHALF, kernel, width, clamp_edges
        );
    }
}

// Portable interior kernel: the generic loop with no edge clamping. Always
// available; the compiler vectorizes it where it can.
template<int KHALF>
auto make_portable_hblur(Accessor2D<const float> input, Accessor2D<float> output, const float* kernel) {
    return simd::Portable<8>([=](const Region2D& region) {
        hblur_region<KHALF>(input, output, kernel, region, /*clamp_edges=*/false);
    });
}

// Scalar edge kernel for the left/right frame (and any alignment remainder),
// where the apron spills past the image and must be clamped.
template<int KHALF>
auto make_hblur_border(Accessor2D<const float> input, Accessor2D<float> output, const float* kernel) {
    return [=](const Region2D& region) {
        hblur_region<KHALF>(input, output, kernel, region, /*clamp_edges=*/true);
    };
}

}  // namespace p10::op

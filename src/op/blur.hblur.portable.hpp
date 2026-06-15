#pragma once

#include <algorithm>
#include <cstdint>

#include <ptensor/span2d.hpp>
#include <p10_internal/simd/tile2d.hpp>

namespace p10::op {

// Horizontal 1D convolution over a tile region. KHALF is the kernel half-size
// (radius), known at compile time so the tap loop unrolls. The interior SIMD
// tiles run with clamp_edges == false (the tiler guarantees the +/-KHALF apron
// is in bounds); the edge frame runs the same loop with clamping.
template<int KHALF>
inline void hblur_region(
    Accessor2D<const float> input,
    Accessor2D<float> output,
    const float* kernel,
    const TileRegion2D& region,
    bool clamp_edges
) {
    const int64_t width = input.cols();
    const int64_t row_end = region.row + region.height;
    const int64_t col_end = region.col + region.width;

    for (int64_t row = region.row; row < row_end; ++row) {
        const float* in_row = input[row].as_span().data();
        float* out_row = output[row].as_span().data();
        for (int64_t col = region.col; col < col_end; ++col) {
            float sum = 0.0F;
            for (int k = -KHALF; k <= KHALF; ++k) {
                int64_t src_col = col + k;
                if (clamp_edges) {
                    src_col = std::clamp<int64_t>(src_col, 0, width - 1);
                }
                sum += in_row[src_col] * kernel[k + KHALF];
            }
            out_row[col] = sum;
        }
    }
}

// Portable interior kernel: the generic loop with no edge clamping. Always
// available; the compiler vectorizes it where it can.
template<int KHALF>
auto make_portable_hblur(Accessor2D<const float> input, Accessor2D<float> output, const float* kernel) {
    return simd::Portable<8>([=](const TileRegion2D& region) {
        hblur_region<KHALF>(input, output, kernel, region, /*clamp_edges=*/false);
    });
}

// Scalar edge kernel for the left/right frame (and any alignment remainder),
// where the apron spills past the image and must be clamped.
template<int KHALF>
auto make_hblur_border(Accessor2D<const float> input, Accessor2D<float> output, const float* kernel) {
    return [=](const TileRegion2D& region) {
        hblur_region<KHALF>(input, output, kernel, region, /*clamp_edges=*/true);
    };
}

}  // namespace p10::op

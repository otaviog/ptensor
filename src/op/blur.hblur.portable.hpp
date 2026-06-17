#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include <p10_internal/simd/tile2d.hpp>
#include <ptensor/span2d.hpp>

#include <type_traits>

namespace p10::op {

// Weighted-sum accumulator for the convolution tap loop. Accumulates in a type
// wide enough for scalar_t (float for small ints, double for >=32-bit ints, the
// float type itself for floats), then on store rounds + saturates back to
// scalar_t for integers and passes floats through unchanged (so the float fast
// path stays bit-identical).
template<typename scalar_t>
struct Accumulator {
    using accum_t = std::conditional_t<
        std::is_floating_point_v<scalar_t>,
        scalar_t,
        std::conditional_t<(sizeof(scalar_t) >= 4), double, float>>;

    accum_t sum = 0;

    void add(scalar_t value, float weight) {
        sum += static_cast<accum_t>(value) * static_cast<accum_t>(weight);
    }

    scalar_t store() const {
        if constexpr (std::is_floating_point_v<scalar_t>) {
            return static_cast<scalar_t>(sum);
        } else {
            using limits = std::numeric_limits<scalar_t>;
            return static_cast<scalar_t>(std::clamp(
                std::round(sum),
                static_cast<accum_t>(limits::min()),
                static_cast<accum_t>(limits::max())
            ));
        }
    }
};

// One horizontal convolution sweep over columns [col_begin, col_end) of a row.
// `half` is the kernel radius; pass a compile-time constant to let the tap loop
// unroll. clamp_edges clamps the apron to [0, width); interior SIMD tiles skip
// it because the tiler guarantees the +/-half apron is in bounds.
template<typename scalar_t>
inline void hblur_scalar(
    Accessor1D<const scalar_t> in_row,
    Accessor1D<scalar_t> out_row,
    int64_t col_begin,
    int64_t col_end,
    int half,
    const float* kernel,
    int64_t width,
    bool clamp_edges
) {
    for (int64_t col = col_begin; col < col_end; ++col) {
        Accumulator<scalar_t> acc;
        for (int k = -half; k <= half; ++k) {
            int64_t src_col = col + k;
            if (clamp_edges) {
                src_col = std::clamp<int64_t>(src_col, 0, width - 1);
            }
            acc.add(in_row[src_col], kernel[k + half]);
        }
        out_row[col] = acc.store();
    }
}

template<typename scalar_t, int64_t KHALF>
inline void hblur_portable(
    Accessor1D<const scalar_t> in_row,
    Accessor1D<scalar_t> out_row,
    int64_t col_begin,
    int64_t col_end,
    const float* kernel
) {
    for (int64_t col = col_begin; col < col_end; ++col) {
        Accumulator<scalar_t> acc;
        for (int k = -KHALF; k <= KHALF; ++k) {
            int64_t src_col = col + k;
            acc.add(in_row[src_col], kernel[k + KHALF]);
        }
        out_row[col] = acc.store();
    }
}


}  // namespace p10::op

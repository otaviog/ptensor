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

// Border sweep over a tile-local row view: `half` is the kernel radius (runtime).
// in_row/out_row are local coordinates (index 0 is the tile's first column); the
// tap clamps into [-left(), right()] so the apron replicates the row edges. Used
// where the tiler cannot guarantee an in-bounds apron.
template<typename scalar_t>
inline void hblur_scalar(
    Accessor1D<const scalar_t> in_row,
    Accessor1D<scalar_t> out_row,
    int half,
    const float* kernel
) {
    const int64_t lo = -in_row.left();
    const int64_t hi = in_row.right();
    for (int64_t col = 0; col < out_row.cols(); ++col) {
        Accumulator<scalar_t> acc;
        for (int k = -half; k <= half; ++k) {
            const int64_t src_col = std::clamp<int64_t>(col + k, lo, hi);
            acc.add(in_row[src_col], kernel[k + half]);
        }
        out_row[col] = acc.store();
    }
}

// Interior sweep over a tile-local row view. KHALF is compile-time so the tap
// loop unrolls. No clamp: the tiler insets the interior by the halo, so the
// +/-KHALF apron is always in bounds (and the row bounds assert it in debug).
template<typename scalar_t, int64_t KHALF>
inline void hblur_portable(
    Accessor1D<const scalar_t> in_row,
    Accessor1D<scalar_t> out_row,
    const float* kernel
) {
    for (int64_t col = 0; col < out_row.cols(); ++col) {
        Accumulator<scalar_t> acc;
        for (int k = -KHALF; k <= KHALF; ++k) {
            acc.add(in_row[col + k], kernel[k + KHALF]);
        }
        out_row[col] = acc.store();
    }
}


}  // namespace p10::op

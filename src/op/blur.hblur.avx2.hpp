#pragma once

#include <p10_internal/simd/compiler.hpp>
#include <p10_internal/simd/tile2d.hpp>
#include <ptensor/span3d.hpp>

#include <type_traits>

#include "blur.hblur.hpp"

namespace p10::op {

// AVX2 horizontal blur spec, tagged float so tile2d only selects it for float.
// The factory is generic over scalar_t so the caller can list it for every
// dtype without an if constexpr (C++ constructs the argument eagerly, before
// tile2d's dispatch runs); the float-only logic lives here, inside the factory:
//
//   * scalar_t == float, intrinsics available -> the real AVX2 kernel. scalar_t
//     is float in this branch, so src/dst are float spans and the tap can use
//     float intrinsics directly.
//   * scalar_t == float, no intrinsics -> empty kernel guarded by a static_assert
//     (it must never be selected on an AVX2-capable target).
//   * scalar_t != float -> empty kernel; tile2d drops it (TargetScalar != scalar_t)
//     and the portable interior runs instead.
//
// SKELETON: the float tap delegates to the portable loop so the pipeline stays
// correct and benchmarkable. Replace it with AVX2 intrinsics: load 8 floats, run
// the +/-KHALF tap loop with _mm256_fmadd_ps, store 8. The region sweep
// (channel/row walk, tile slicing, transposed write) is shared and stays.
template<typename scalar_t, int64_t KHALF>
auto make_avx2_hblur(Span3D<const scalar_t> src, Span3D<scalar_t> dst, const float* kernel) {
    if constexpr (std::is_same_v<scalar_t, float>) {
#if PTENSOR_HAS_INTRINSICS_H
        return simd::Avx2<8, float>(hblur_region_sweep<float>(
            src,
            dst,
            kernel,
            [](Accessor1D<const float> in_row, Accessor1D<float> out_row, const float* k) {
                // TODO: replace with AVX2 intrinsics (8-wide float, _mm256_fmadd_ps).
                hblur_portable<float, KHALF>(in_row, out_row, k);
            }
        ));
#else
        (void)src;
        (void)dst;
        (void)kernel;
        return simd::Avx2<8, float>([](const Region2D&) {
            static_assert(
                !simd::is_compiler_supported(simd::SimdSet::AVX2),
                "empty AVX2 hblur kernel instantiated on an AVX2-capable target"
            );
        });
#endif
    } else {
        (void)src;
        (void)dst;
        (void)kernel;
        // No integer AVX2 path; tile2d never selects a float-tagged kernel here.
        return simd::Avx2<8, float>([](const Region2D&) {});
    }
}

}  // namespace p10::op

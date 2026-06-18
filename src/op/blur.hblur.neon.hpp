#pragma once

#include <p10_internal/simd/compiler.hpp>
#include <p10_internal/simd/tile2d.hpp>
#include <ptensor/span3d.hpp>

#include <type_traits>

#include "blur.hblur.hpp"

namespace p10::op {

// NEON horizontal blur spec, tagged float so tile2d only selects it for float.
// The factory is generic over scalar_t so the caller can list it for every dtype
// without an if constexpr (the argument is constructed eagerly, before tile2d's
// dispatch); the float-only logic lives here. See make_avx2_hblur for the three
// branches and the rationale.
//
// SKELETON: the float tap delegates to the portable loop. Replace it with NEON
// intrinsics: load 4 floats (float32x4_t), run the +/-KHALF tap loop with
// vfmaq_f32, store 4 (an 8-wide tile is two 4-lane vectors). The region sweep is
// shared and stays.
template<typename scalar_t, int64_t KHALF>
auto make_neon_hblur(Span3D<const scalar_t> src, Span3D<scalar_t> dst, const float* kernel) {
    if constexpr (std::is_same_v<scalar_t, float>) {
#if PTENSOR_HAS_NEON
        return simd::Neon<8, float>(hblur_region_sweep<float>(
            src,
            dst,
            kernel,
            [](Accessor1D<const float> in_row, Accessor1D<float> out_row, const float* k) {
                // TODO: replace with NEON intrinsics (float32x4_t, vfmaq_f32).
                hblur_portable<float, KHALF>(in_row, out_row, k);
            }
        ));
#else
        (void)src;
        (void)dst;
        (void)kernel;
        return simd::Neon<8, float>([](const Region2D&) {
            static_assert(
                !simd::is_compiler_supported(simd::SimdSet::AdvSIMD),
                "empty NEON hblur kernel instantiated on a NEON-capable target"
            );
        });
#endif
    } else {
        (void)src;
        (void)dst;
        (void)kernel;
        // No integer NEON path; tile2d never selects a float-tagged kernel here.
        return simd::Neon<8, float>([](const Region2D&) {});
    }
}

}  // namespace p10::op

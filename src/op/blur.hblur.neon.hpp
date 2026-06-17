#pragma once

#include <ptensor/accessor2d.hpp>
#include <p10_internal/simd/compiler.hpp>
#include <p10_internal/simd/tile2d.hpp>

#include "blur.hblur.portable.hpp"

namespace p10::op {

// NEON horizontal blur kernel for float32. With NEON it returns a real kernel;
// otherwise an empty kernel that tile2d compiles out (is_compiler_supported is
// false off aarch64).
//
// SKELETON: the body currently delegates to the portable loop so the blur
// pipeline stays correct and benchmarkable. Replace it with NEON intrinsics:
// load 4 floats (float32x4_t), run the KHALF tap loop with vfmaq_f32, store 4
// (an 8-wide tile is two 4-lane vectors).
inline auto make_neon_hblur(
    Accessor2D<const float> input,
    Accessor2D<float> output,
    const float* kernel,
    int half
) {

    (void) input;
    (void) output;
    (void) kernel;
    (void) half;
    return simd::Neon<8, float>([](const Region2D&) {
        // static_assert(
        //!simd::is_compiler_supported(simd::SimdSet::AdvSIMD),
        //"empty NEON hblur kernel instantiated on a NEON-capable target"
        //);
    });
}

}  // namespace p10::op

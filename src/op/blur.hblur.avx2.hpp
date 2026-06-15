#pragma once

#include <cstdint>

#include <ptensor/span2d.hpp>
#include <p10_internal/simd/compiler.hpp>
#include <p10_internal/simd/tile2d.hpp>

#include "blur.hblur.portable.hpp"

namespace p10::op {

// AVX2 horizontal blur kernel for float32. With intrinsics it returns a real
// AVX2 kernel; without them an empty kernel that tile2d compiles out
// (is_compiler_supported(AVX2) is false off x86).
//
// SKELETON: the body currently delegates to the portable loop so the blur
// pipeline stays correct and benchmarkable. Replace it with AVX2 intrinsics:
// load 8 floats, run the KHALF tap loop with _mm256_fmadd_ps, store 8.
template<int KHALF>
auto make_avx2_hblur(Accessor2D<const float> input, Accessor2D<float> output, const float* kernel) {
#if PTENSOR_HAS_INTRINSICS_H
    return simd::Avx2<8>([=](const Region2D& region) {
        // TODO: replace with AVX2 intrinsics (8-wide float, _mm256_fmadd_ps).
        hblur_region<KHALF>(input, output, kernel, region, /*clamp_edges=*/false);
    });
#else
    (void) input;
    (void) output;
    (void) kernel;
    return simd::Avx2<8>([](const Region2D&) {
        static_assert(
            !simd::is_compiler_supported(simd::SimdSet::AVX2),
            "empty AVX2 hblur kernel instantiated on an AVX2-capable target"
        );
    });
#endif
}

}  // namespace p10::op

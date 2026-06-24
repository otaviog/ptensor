#pragma once

#include <cstdint>

#include <p10_internal/simd/compiler.hpp>
#include <p10_internal/simd/tile2d.hpp>

#if PTENSOR_HAS_NEON
    #include <arm_neon.h>
#endif

namespace p10 {

#if PTENSOR_HAS_NEON

// Transpose a 4x4 block of 32-bit elements in NEON registers (128-bit = 4 lanes).
inline void
transpose_neon_4x4_32(int32_t const* src, int64_t src_stride, int32_t* dst, int64_t dst_stride) {
    int32x4_t r0 = vld1q_s32(src + 0 * src_stride);
    int32x4_t r1 = vld1q_s32(src + 1 * src_stride);
    int32x4_t r2 = vld1q_s32(src + 2 * src_stride);
    int32x4_t r3 = vld1q_s32(src + 3 * src_stride);

    // Transpose adjacent 32-bit lane pairs, then swap the 64-bit halves.
    int32x4x2_t t01 = vtrnq_s32(r0, r1);
    int32x4x2_t t23 = vtrnq_s32(r2, r3);

    int32x4_t o0 = vcombine_s32(vget_low_s32(t01.val[0]), vget_low_s32(t23.val[0]));
    int32x4_t o1 = vcombine_s32(vget_low_s32(t01.val[1]), vget_low_s32(t23.val[1]));
    int32x4_t o2 = vcombine_s32(vget_high_s32(t01.val[0]), vget_high_s32(t23.val[0]));
    int32x4_t o3 = vcombine_s32(vget_high_s32(t01.val[1]), vget_high_s32(t23.val[1]));

    vst1q_s32(dst + 0 * dst_stride, o0);
    vst1q_s32(dst + 1 * dst_stride, o1);
    vst1q_s32(dst + 2 * dst_stride, o2);
    vst1q_s32(dst + 3 * dst_stride, o3);
}

// Transpose an 8x8 block as four transposed 4x4 quadrants, with the
// off-diagonal quadrants swapped: tile (r, c) maps to (c, r).
inline void
transpose_neon_8x8_32(int32_t const* src, int64_t src_stride, int32_t* dst, int64_t dst_stride) {
    transpose_neon_4x4_32(src, src_stride, dst, dst_stride);
    transpose_neon_4x4_32(src + 4, src_stride, dst + 4 * dst_stride, dst_stride);
    transpose_neon_4x4_32(src + 4 * src_stride, src_stride, dst + 4, dst_stride);
    transpose_neon_4x4_32(
        src + 4 * src_stride + 4,
        src_stride,
        dst + 4 * dst_stride + 4,
        dst_stride
    );
}
#endif  // PTENSOR_HAS_NEON

// Build a NEON 8x8 transpose kernel for 32-bit elements. With NEON the kernel
// transposes a register tile; otherwise it returns an empty kernel that tile2d's
// dispatch compiles out (is_compiler_supported(AdvSIMD) is false off aarch64).
template<size_t SIMD_BLOCK, typename ScalarT, typename SrcBlock, typename DstBlock>
auto make_neon_transpose(
    SrcBlock src_block,
    DstBlock dst_block,
    int64_t src_stride,
    int64_t dst_stride
) {
#if PTENSOR_HAS_NEON
    return simd::Neon<SIMD_BLOCK, ScalarT>([=](const Region2D& region) {
        transpose_neon_8x8_32(
            reinterpret_cast<const int32_t*>(src_block(region)),
            src_stride,
            reinterpret_cast<int32_t*>(dst_block(region)),
            dst_stride
        );
    });
#else
    (void)src_block;
    (void)dst_block;
    (void)src_stride;
    (void)dst_stride;
    return simd::Neon<SIMD_BLOCK, ScalarT>([](const Region2D&) {
        // This no-op body must never reach codegen on a NEON-capable target,
        // where tile2d would select it and silently skip the transpose.
        static_assert(
            !simd::is_compiler_supported(simd::SimdSet::AdvSIMD),
            "empty NEON transpose kernel instantiated on a NEON-capable target"
        );
    });
#endif
}

}  // namespace p10

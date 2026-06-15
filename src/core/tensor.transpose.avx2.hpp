#pragma once

#include <cstdint>

#include <p10_internal/simd/compiler.hpp>
#include <p10_internal/simd/tile2d.hpp>

#if PTENSOR_HAS_INTRINSICS_H
    #include <immintrin.h>
#endif

namespace p10 {

#if PTENSOR_HAS_INTRINSICS_H

// Transpose an 8x8 block of 32-bit elements entirely in AVX2 registers.
PTENSOR_AVX2 inline void
transpose_avx2_8x8_32(int32_t const* src, int64_t src_stride, int32_t* dst, int64_t dst_stride) {
    __m256i row0 = _mm256_loadu_si256((__m256i const*)src);
    __m256i row1 = _mm256_loadu_si256((__m256i const*)(src + src_stride));
    __m256i row2 = _mm256_loadu_si256((__m256i const*)(src + 2 * src_stride));
    __m256i row3 = _mm256_loadu_si256((__m256i const*)(src + 3 * src_stride));
    __m256i row4 = _mm256_loadu_si256((__m256i const*)(src + 4 * src_stride));
    __m256i row5 = _mm256_loadu_si256((__m256i const*)(src + 5 * src_stride));
    __m256i row6 = _mm256_loadu_si256((__m256i const*)(src + 6 * src_stride));
    __m256i row7 = _mm256_loadu_si256((__m256i const*)(src + 7 * src_stride));
    /* Starts with
       r0 = 00 01 02 03 | 04 05 06 07
       r1 = 08 09 10 11 | 12 13 14 15
       r2 = 16 17 18 19 | 20 21 22 23
       r3 = 24 25 26 27 | 28 29 30 31
       r4 = 32 33 34 35 | 36 37 38 39
       r5 = 40 41 42 43 | 44 45 46 47
       r6 = 48 49 50 51 | 52 53 54 55
       r7 = 56 57 58 59 | 60 61 62 63
    */

    __m256i t0a = _mm256_unpacklo_epi32(row0, row1);
    __m256i t0b = _mm256_unpackhi_epi32(row0, row1);

    __m256i t2a = _mm256_unpacklo_epi32(row2, row3);
    __m256i t2b = _mm256_unpackhi_epi32(row2, row3);

    __m256i t4a = _mm256_unpacklo_epi32(row4, row5);
    __m256i t4b = _mm256_unpackhi_epi32(row4, row5);

    __m256i t6a = _mm256_unpacklo_epi32(row6, row7);
    __m256i t6b = _mm256_unpackhi_epi32(row6, row7);
    /* Transposed the first 2x2 matrices
       t0a = [00 08; 01 09] | [04 12; 05 13]
       t0b = [02 10; 03 11] | [06 14; 07 15]

       t2a = [16 24; 17 25] | [20 28; 21 29]
       t2b = [18 26; 19 27] | [22 30; 23 31]

       t4a = [32 40; 33 41] | [36 44; 37 45]
       t4b = [34 42; 35 43] | [38 46; 39 47]

       t6a = [48 56; 49 57] | [52 60; 53 61]
       t6b = [50 58; 51 59] | [54 62; 55 63]
    */

    __m256i s0 = _mm256_unpacklo_epi64(t0a, t2a);
    __m256i s1 = _mm256_unpackhi_epi64(t0a, t2a);

    __m256i s2 = _mm256_unpacklo_epi64(t0b, t2b);
    __m256i s3 = _mm256_unpackhi_epi64(t0b, t2b);

    __m256i s4 = _mm256_unpacklo_epi64(t4a, t6a);
    __m256i s5 = _mm256_unpackhi_epi64(t4a, t6a);

    __m256i s6 = _mm256_unpacklo_epi64(t4b, t6b);
    __m256i s7 = _mm256_unpackhi_epi64(t4b, t6b);

    /* Now unites the 4x4 blocks into 8x8 blocks
       s0 = [00 08 16 24] | [04 12 20 28]
       s1 = [01 09 17 25] | [05 13 21 29]

       s2 = [02 10 18 26] | [06 14 22 30]
       s3 = [03 11 19 27] | [07 15 23 31]

       s4 = [32 40 48 56] | [36 44 52 60]
       s5 = [33 41 49 57] | [37 45 53 61]

       s6 = [34 42 50 58] | [38 46 54 62]
       s7 = [35 43 51 59] | [39 47 55 63]
    */

    constexpr int PERMUTE_MASK_LOW_128bits = 0x20;
    constexpr int PERMUTE_MASK_HIGH_128bits = 0x31;
    __m256i row0t = _mm256_permute2f128_si256(s0, s4, PERMUTE_MASK_LOW_128bits);
    __m256i row1t = _mm256_permute2f128_si256(s1, s5, PERMUTE_MASK_LOW_128bits);
    __m256i row2t = _mm256_permute2f128_si256(s2, s6, PERMUTE_MASK_LOW_128bits);
    __m256i row3t = _mm256_permute2f128_si256(s3, s7, PERMUTE_MASK_LOW_128bits);
    __m256i row4t = _mm256_permute2f128_si256(s0, s4, PERMUTE_MASK_HIGH_128bits);
    __m256i row5t = _mm256_permute2f128_si256(s1, s5, PERMUTE_MASK_HIGH_128bits);
    __m256i row6t = _mm256_permute2f128_si256(s2, s6, PERMUTE_MASK_HIGH_128bits);
    __m256i row7t = _mm256_permute2f128_si256(s3, s7, PERMUTE_MASK_HIGH_128bits);

    _mm256_storeu_si256((__m256i*)(dst + 0 * dst_stride), row0t);
    _mm256_storeu_si256((__m256i*)(dst + 1 * dst_stride), row1t);
    _mm256_storeu_si256((__m256i*)(dst + 2 * dst_stride), row2t);
    _mm256_storeu_si256((__m256i*)(dst + 3 * dst_stride), row3t);
    _mm256_storeu_si256((__m256i*)(dst + 4 * dst_stride), row4t);
    _mm256_storeu_si256((__m256i*)(dst + 5 * dst_stride), row5t);
    _mm256_storeu_si256((__m256i*)(dst + 6 * dst_stride), row6t);
    _mm256_storeu_si256((__m256i*)(dst + 7 * dst_stride), row7t);
}
#endif  // PTENSOR_HAS_INTRINSICS_H

// Build an AVX2 8x8 transpose kernel for 32-bit elements. With intrinsics the
// kernel transposes a register tile; without them it returns an empty kernel
// that tile2d's dispatch compiles out (is_compiler_supported(AVX2) is false on
// non-x86 targets, so it is never selected).
template<size_t SIMD_BLOCK, typename SrcBlock, typename DstBlock>
auto make_avx2_transpose(
    SrcBlock src_block,
    DstBlock dst_block,
    int64_t src_stride,
    int64_t dst_stride
) {
#if PTENSOR_HAS_INTRINSICS_H
    return simd::Avx2<SIMD_BLOCK>([=](const Region2D& region) {
        transpose_avx2_8x8_32(
            reinterpret_cast<const int32_t*>(src_block(region)),
            src_stride,
            reinterpret_cast<int32_t*>(dst_block(region)),
            dst_stride
        );
    });
#else
    (void) src_block;
    (void) dst_block;
    (void) src_stride;
    (void) dst_stride;
    return simd::Avx2<SIMD_BLOCK>([](const Region2D&) {
        // This no-op body must never reach codegen on an AVX2-capable target,
        // where tile2d would select it and silently skip the transpose.
        static_assert(
            !simd::is_compiler_supported(simd::SimdSet::AVX2),
            "empty AVX2 transpose kernel instantiated on an AVX2-capable target"
        );
    });
#endif
}

}  // namespace p10

#include "tensor.hpp"

#include <cstdint>
#include <iostream>

#include <immintrin.h>

#include "cpuid/cpuid.hpp"
#include "p10_error.hpp"
#include "tensor_print.hpp"

namespace p10 {
template<typename scalar_t>
void transpose_generic(
    int64_t rows,
    int64_t cols,
    const scalar_t* src,
    size_t src_stride,
    scalar_t* dst,
    size_t dst_stride
) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            dst[j * dst_stride + i] = src[i * src_stride + j];
        }
    }
}

template<typename scalar_t>
void transpose_8x8_generic(
    const scalar_t* src,
    size_t src_stride,
    scalar_t* dst,
    size_t dst_stride
) {
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            dst[j * dst_stride + i] = src[i * src_stride + j];
        }
    }
}

__attribute__((target("avx2"))) void
transpose_avx2_8x8_32(int32_t const* src, size_t src_stride, int32_t* dst, size_t dst_stride) {
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

P10Error Tensor::transpose(Tensor& other) const {
    if (blob_.device() != Device::Cpu) {
        return P10Error::NotImplemented << "Transpose is only implemented for CPU tensors";
    }

    return visit([this, &other](auto type_span) -> P10Error {
        using scalar_t = std::remove_const_t<typename decltype(type_span)::element_type>;

        auto src_span_res = this->as_span2d<const scalar_t>();
        if (src_span_res.is_error()) {
            return src_span_res.error();
        }
        auto src_span = src_span_res.unwrap();
        P10_RETURN_IF_ERROR(other.create(make_shape(src_span.width(), src_span.height()), dtype()));
        auto dest_span = other.as_span2d<scalar_t>().unwrap();

        const size_t rows = src_span.height();
        const size_t cols = src_span.width();

        const size_t src_stride = size_t(src_span.width());
        const size_t dst_stride = size_t(dest_span.width());

        if (rows < 8 || cols < 8) {
            // Fallback to generic transpose for small tensors
            transpose_generic(
                rows,
                cols,
                src_span.row(0),
                src_stride,
                dest_span.row(0),
                dst_stride
            );
            return P10Error::Ok;
        }

        const size_t aligned_max_rows = rows - (rows % 8);
        const size_t aligned_max_cols = cols - (cols % 8);

        for (size_t r = 0; r < aligned_max_rows; r += 8) {
            const auto src_row = src_span.row(r);
            for (size_t c = 0; c < aligned_max_cols; c += 8) {
                const scalar_t* src_block = &src_row[c];
                scalar_t* dest_block = &dest_span.row(c)[r];

                if constexpr (sizeof(scalar_t) == sizeof(int32_t)) {
                    if (is_avx2_supported()) {
                        transpose_avx2_8x8_32(
                            reinterpret_cast<const int32_t*>(&src_row[c]),
                            src_stride,
                            reinterpret_cast<int32_t*>(&dest_span.row(c)[r]),
                            dst_stride
                        );
                        continue;
                    }
                }

                transpose_8x8_generic(src_block, src_stride, dest_block, dst_stride);
            }
            for (size_t rr = r; rr < r + 8; rr++) {
                const auto src_row = src_span.row(rr);
                for (size_t c = aligned_max_cols; c < cols; c++) {
                    dest_span.row(c)[rr] = src_row[c];
                }
            }
        }

        for (size_t r = aligned_max_rows; r < rows; r++) {
            const auto src_row = src_span.row(r);
            for (size_t c = 0; c < cols; c++) {
                dest_span.row(c)[r] = src_row[c];
            }
        }
        return P10Error::Ok;
    });
}
}  // namespace p10
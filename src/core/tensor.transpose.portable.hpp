#pragma once

#include <cstdint>

#include <p10_internal/simd/tile2d.hpp>

namespace p10 {

// Transpose a rows x cols rectangle element by element. Used as the scalar
// border kernel for the leftover edges a SIMD tile cannot cover.
template<typename ScalarT>
inline void transpose_generic(
    int64_t rows,
    int64_t cols,
    const ScalarT* src,
    int64_t src_stride,
    ScalarT* dst,
    int64_t dst_stride
) {
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            dst[(j * dst_stride) + i] = src[(i * src_stride) + j];
        }
    }
}

// Transpose a fixed 8x8 block element by element. Portable interior kernel; the
// compiler vectorizes it where it can, otherwise it stays scalar.
template<typename ScalarT>
inline void
transpose_8x8_generic(const ScalarT* src, int64_t src_stride, ScalarT* dst, int64_t dst_stride) {
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            dst[(j * dst_stride) + i] = src[(i * src_stride) + j];
        }
    }
}

// Build the portable 8x8 transpose kernel. Always available on every target.
template<size_t SIMD_BLOCK, typename ScalarT, typename SrcBlock, typename DstBlock>
auto make_portable_transpose(
    SrcBlock src_block,
    DstBlock dst_block,
    int64_t src_stride,
    int64_t dst_stride
) {
    return simd::Portable<SIMD_BLOCK, ScalarT>([=](const Region2D& region) {
        transpose_8x8_generic<ScalarT>(src_block(region), src_stride, dst_block(region), dst_stride);
    });
}

// Build the scalar border kernel for the leftover rectangles (not a SIMD spec,
// a plain callable that tile2d/tile2d_autoblock takes as the scalar fallback).
template<typename ScalarT, typename SrcBlock, typename DstBlock>
auto make_transpose_border(
    SrcBlock src_block,
    DstBlock dst_block,
    int64_t src_stride,
    int64_t dst_stride
) {
    return [=](const Region2D& region) {
        transpose_generic<ScalarT>(
            region.height,
            region.width,
            src_block(region),
            src_stride,
            dst_block(region),
            dst_stride
        );
    };
}

}  // namespace p10

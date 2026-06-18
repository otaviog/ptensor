#include "tensor.hpp"

#include <cstdint>
#include <utility>

#include <p10_internal/simd/tile2d.hpp>

#include "p10_error.hpp"
#include "tensor.transpose.avx2.hpp"
#include "tensor.transpose.neon.hpp"
#include "tensor.transpose.portable.hpp"

namespace p10 {

P10Error Tensor::transpose(Tensor& other) const {
    if (blob_.device() != Device::Cpu) {
        return P10Error::NotImplemented << "Transpose is only implemented for CPU tensors";
    }

    if (!is_contiguous()) {
        return P10Error::NotImplemented << "Transpose is only implemented for contiguous tensors";
    }

    // Transposing into self would realloc this tensor's blob while the kernels
    // still read its old storage. Route through a temporary, then move it back.
    if (&other == this) {
        Tensor result;
        P10_RETURN_IF_ERROR(transpose(result));
        other = std::move(result);
        return P10Error::Ok;
    }

    return visit([this, &other](auto type_span) -> P10Error {
        using ScalarT = std::remove_const_t<typename decltype(type_span)::element_type>;

        auto src_span_res = this->as_span2d<const ScalarT>();
        if (src_span_res.is_error()) {
            return src_span_res.error();
        }
        auto src_span = src_span_res.unwrap();
        P10_RETURN_IF_ERROR(other.create(make_shape(src_span.cols(), src_span.rows()), dtype()));
        auto dest_span_res = other.as_span2d<ScalarT>();
        if (dest_span_res.is_error()) {
            return dest_span_res.error();
        }
        auto dest_span = dest_span_res.unwrap();

        const int64_t rows = src_span.rows();
        const int64_t cols = src_span.cols();

        const int64_t src_stride = src_span.cols();
        const int64_t dst_stride = dest_span.cols();

        constexpr size_t SIMD_BLOCK = 8;

        // A src tile at (row, col) transposes into the dst tile at (col, row).
        const auto src_block = [&](const Region2D& region) {
            return &src_span[region.row][region.col];
        };
        const auto dst_block = [&](const Region2D& region) {
            return &dest_span[region.col][region.row];
        };

        auto edge = make_transpose_border<ScalarT>(src_block, dst_block, src_stride, dst_stride);
        auto portable = make_portable_transpose<SIMD_BLOCK, ScalarT>(
            src_block,
            dst_block,
            src_stride,
            dst_stride
        );

        // tile2d picks the first kernel the running CPU supports (via cpuid),
        // otherwise the next, and the edge kernel always handles the borders.
        // The SIMD 8x8 kernels move 32-bit lanes, so they also serve float32
        // (the bit pattern is shuffled untouched); larger types fall to scalar.
        // Transpose has no stencil halo, so the tile border is empty.
        if constexpr (sizeof(ScalarT) == sizeof(int32_t)) {
            simd::tile2d<ScalarT>(
                rows,
                cols,
                simd::TileBorder {},
                edge,
                make_avx2_transpose<SIMD_BLOCK, ScalarT>(
                    src_block,
                    dst_block,
                    src_stride,
                    dst_stride
                ),
                make_neon_transpose<SIMD_BLOCK, ScalarT>(
                    src_block,
                    dst_block,
                    src_stride,
                    dst_stride
                ),
                portable
            );
            return P10Error::Ok;
        }
        simd::tile2d<ScalarT>(rows, cols, simd::TileBorder {}, edge, portable);
        return P10Error::Ok;
    });
}
}  // namespace p10
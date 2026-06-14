#include "tensor.hpp"

#include <cstdint>
#include <utility>

#include <p10_internal/simd/tile.hpp>

#include "p10_error.hpp"
#include "tensor.transpose.avx2.hpp"
#include "tensor.transpose.neon.hpp"

namespace p10 {
namespace {

    template<typename ScalarT>
    void transpose_generic(
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

    template<typename ScalarT>
    void
    transpose_8x8_generic(const ScalarT* src, int64_t src_stride, ScalarT* dst, int64_t dst_stride) {
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                dst[(j * dst_stride) + i] = src[(i * src_stride) + j];
            }
        }
    }

}  // namespace

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
        P10_RETURN_IF_ERROR(other.create(make_shape(src_span.width(), src_span.height()), dtype()));
        auto dest_span_res = other.as_span2d<ScalarT>();
        if (dest_span_res.is_error()) {
            return dest_span_res.error();
        }
        auto dest_span = dest_span_res.unwrap();

        const int64_t rows = src_span.height();
        const int64_t cols = src_span.width();

        const int64_t src_stride = src_span.width();
        const int64_t dst_stride = dest_span.width();

        constexpr size_t SIMD_BLOCK = 8;

        // A src tile at (row, col) transposes into the dst tile at (col, row).
        const auto src_block = [&](const TileRegion2D& region) {
            return &src_span.row(region.row)[region.col];
        };
        const auto dst_block = [&](const TileRegion2D& region) {
            return &dest_span.row(region.col)[region.row];
        };

        // Interior: full SIMD_BLOCK x SIMD_BLOCK tiles, transposed in registers.
        const auto transpose_block = [&](const TileRegion2D& region) {
            transpose_8x8_generic(src_block(region), src_stride, dst_block(region), dst_stride);
        };

        // Borders: any leftover rectangle, transposed element by element.
        const auto scalar_impl = [&](const TileRegion2D& region) {
            transpose_generic<ScalarT>(
                region.height,
                region.width,
                src_block(region),
                src_stride,
                dst_block(region),
                dst_stride
            );
        };

        // tile2d picks the first kernel the running CPU supports (via cpuid),
        // otherwise the next, and the scalar kernel always handles the borders.
        // The SIMD 8x8 kernels move 32-bit lanes, so they also serve float32
        // (the bit pattern is shuffled untouched); larger types fall to scalar.
        if constexpr (sizeof(ScalarT) == sizeof(int32_t)) {
            simd::tile2d<ScalarT>(
                rows,
                cols,
                scalar_impl,
                make_avx2_transpose<SIMD_BLOCK>(src_block, dst_block, src_stride, dst_stride),
                make_neon_transpose<SIMD_BLOCK>(src_block, dst_block, src_stride, dst_stride),
                simd::Portable<SIMD_BLOCK>(transpose_block)
            );
            return P10Error::Ok;
        }
        simd::tile2d<ScalarT>(
            rows, cols, scalar_impl, simd::Portable<SIMD_BLOCK>(transpose_block)
        );
        return P10Error::Ok;
    });
}
}  // namespace p10
#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <utility>

#include <ptensor/region2d.hpp>

#include "bitwise_math.hpp"
#include "cpuid.hpp"
#include "tile_execution.hpp"

namespace p10::simd {

template<typename F>
concept TileKernel2DFn = std::invocable<F, Region2D>;

// Runtime halo size for stencil kernels (blur, convolution, ...). Each SIMD
// tile reads `horizontal` extra columns left/right and `vertical` extra rows
// top/bottom; the tiler keeps the SIMD kernel inside the region where that apron
// stays in bounds and hands the edge frame to the scalar kernel (which clamps).
// A power-of-two halo is not required; only the cache block must be. Default {}
// (no halo) reproduces a plain elementwise tiling. Passed as a runtime value so
// a single kernel instantiation serves every kernel size.
struct TileBorder {
    int64_t horizontal = 0;
    int64_t vertical = 0;
};

template<
    size_t CACHE_BLOCK,
    size_t SIMD_BLOCK,
    TileExecution ExecutionMode,
    TileKernel2DFn SimdFn,
    TileKernel2DFn ScalarFn>
void tile2d_blocked(
    int64_t rows,
    int64_t cols,
    TileBorder border,
    SimdFn&& simd_impl,
    ScalarFn&& scalar_impl
) {
    static_assert(CACHE_BLOCK % SIMD_BLOCK == 0, "CACHE_BLOCK must be a multiple of SIMD_BLOCK");

    constexpr int64_t CACHE = CACHE_BLOCK;
    constexpr int64_t SIMD = SIMD_BLOCK;
    const int64_t BORDER_V = border.vertical;
    const int64_t BORDER_H = border.horizontal;

    // SIMD walks the interior inset by the halo, aligned down to whole cache
    // blocks; everything outside is the edge frame plus the alignment remainder.
    const int64_t interior_rows = rows - (2 * BORDER_V);
    const int64_t interior_cols = cols - (2 * BORDER_H);
    const int64_t tiled_rows =
        interior_rows > 0 ? interior_rows - bitwise_modulo<CACHE_BLOCK>(interior_rows) : 0;
    const int64_t tiled_cols =
        interior_cols > 0 ? interior_cols - bitwise_modulo<CACHE_BLOCK>(interior_cols) : 0;

    const int64_t row_begin = BORDER_V;
    const int64_t col_begin = BORDER_H;
    const int64_t row_end = row_begin + tiled_rows;  // first row past the SIMD area
    const int64_t col_end = col_begin + tiled_cols;  // first col past the SIMD area

    // Interior [row_begin, row_end) x [col_begin, col_end), in cache blocks that
    // are further split into SIMD_BLOCK x SIMD_BLOCK tiles.
#pragma omp parallel for collapse(2) schedule(static) if (ExecutionMode == TileExecution::PARALLEL)
    for (int64_t block_row = row_begin; block_row < row_end; block_row += CACHE) {
        for (int64_t block_col = col_begin; block_col < col_end; block_col += CACHE) {
            for (int64_t simd_row = block_row; simd_row < block_row + CACHE; simd_row += SIMD) {
                for (int64_t simd_col = block_col; simd_col < block_col + CACHE; simd_col += SIMD) {
                    simd_impl(
                        Region2D {.row = simd_row, .col = simd_col, .height = SIMD, .width = SIMD}
                    );
                }
            }
        }
    }

    // Edge frame + alignment remainder, as four non-overlapping scalar rectangles
    // around the SIMD area. With Border{} this collapses to the right and bottom
    // remainder bands of a plain tiling.
    if (row_begin > 0) {  // top band
        scalar_impl(Region2D {.row = 0, .col = 0, .height = row_begin, .width = cols});
    }
    if (row_end < rows) {  // bottom band
        scalar_impl(Region2D {.row = row_end, .col = 0, .height = rows - row_end, .width = cols});
    }
    if (tiled_rows > 0 && col_begin > 0) {  // left band
        scalar_impl(
            Region2D {.row = row_begin, .col = 0, .height = tiled_rows, .width = col_begin}
        );
    }
    if (tiled_rows > 0 && col_end < cols) {  // right band
        scalar_impl(
            Region2D {
                .row = row_begin,
                .col = col_end,
                .height = tiled_rows,
                .width = cols - col_end
            }
        );
    }
}

// Round a target cache-tile side (in elements) down to a multiple of
// SIMD_BLOCK, never below one block. Guarantees tile2d_blocked's
// CACHE % SIMD_BLOCK == 0 invariant for any SIMD_BLOCK, so the divisibility
// static_assert can never fire from an auto-picked bucket.
template<size_t SIMD_BLOCK>
constexpr size_t floor_to_simd(size_t target) {
    const size_t blocks = target / SIMD_BLOCK;
    return (blocks == 0 ? size_t {1} : blocks) * SIMD_BLOCK;
}

template<
    size_t SIMD_BLOCK,
    typename scalar_t,
    TileExecution ExecutionMode = TileExecution::SEQUENTIAL,
    TileKernel2DFn SimdFn,
    TileKernel2DFn ScalarFn>
void tile2d_autoblock(
    int64_t rows,
    int64_t cols,
    TileBorder border,
    SimdFn&& simd_impl,
    ScalarFn&& scalar_impl
) {
    // Side of a square cache tile, in elements: ~sqrt(L1) bytes / element size,
    // so the working set stays in L1d (the src and dst tiles share L1).
    const int64_t l1_tile_side =
        static_cast<int64_t>(std::sqrt(l1_cache_size())) / static_cast<int64_t>(sizeof(scalar_t));

    if (std::max(rows, cols) < l1_tile_side) {
        std::forward<ScalarFn>(scalar_impl)(
            Region2D {.row = 0, .col = 0, .height = rows, .width = cols}
        );
        return;
    }

    if (l1_tile_side >= 1024) {
        tile2d_blocked<floor_to_simd<SIMD_BLOCK>(1024), SIMD_BLOCK, ExecutionMode>(
            rows,
            cols,
            border,
            std::forward<SimdFn>(simd_impl),
            std::forward<ScalarFn>(scalar_impl)
        );
    } else if (l1_tile_side >= 512) {
        tile2d_blocked<floor_to_simd<SIMD_BLOCK>(512), SIMD_BLOCK, ExecutionMode>(
            rows,
            cols,
            border,
            std::forward<SimdFn>(simd_impl),
            std::forward<ScalarFn>(scalar_impl)
        );
    } else if (l1_tile_side >= 256) {
        tile2d_blocked<floor_to_simd<SIMD_BLOCK>(256), SIMD_BLOCK, ExecutionMode>(
            rows,
            cols,
            border,
            std::forward<SimdFn>(simd_impl),
            std::forward<ScalarFn>(scalar_impl)
        );
    } else if (l1_tile_side >= 128) {
        tile2d_blocked<floor_to_simd<SIMD_BLOCK>(128), SIMD_BLOCK, ExecutionMode>(
            rows,
            cols,
            border,
            std::forward<SimdFn>(simd_impl),
            std::forward<ScalarFn>(scalar_impl)
        );
    } else if (l1_tile_side >= 64) {
        tile2d_blocked<floor_to_simd<SIMD_BLOCK>(64), SIMD_BLOCK, ExecutionMode>(
            rows,
            cols,
            border,
            std::forward<SimdFn>(simd_impl),
            std::forward<ScalarFn>(scalar_impl)
        );
    } else {
        tile2d_blocked<floor_to_simd<SIMD_BLOCK>(32), SIMD_BLOCK, ExecutionMode>(
            rows,
            cols,
            border,
            std::forward<SimdFn>(simd_impl),
            std::forward<ScalarFn>(scalar_impl)
        );
    }
}

template<size_t SimdBlock, SimdSet TargetInstructions, typename TargetType, TileKernel2DFn KernelFn>
struct TileKernel2D {
    static constexpr size_t SIMD_BLOCK = SimdBlock;
    static constexpr SimdSet INSTRUCTIONS = TargetInstructions;
    using TargetScalar = TargetType;
    KernelFn fn;
};

template<typename T>
concept TileKernelSpec2D = requires {
    { T::SIMD_BLOCK } -> std::convertible_to<size_t>;
    { T::INSTRUCTIONS } -> std::convertible_to<SimdSet>;
    typename T::TargetScalar;
} && TileKernel2DFn<decltype(T::fn)>;

template<
    typename scalar_t,
    TileExecution ExecutionMode = TileExecution::SEQUENTIAL,
    TileKernel2DFn ScalarKn>
void tile2d(int64_t rows, int64_t cols, TileBorder border, ScalarKn&& scalar_impl) {
    (void)border;
    scalar_impl(Region2D {.row = 0, .col = 0, .height = rows, .width = cols});
}

template<
    typename scalar_t,
    TileExecution ExecutionMode = TileExecution::SEQUENTIAL,
    TileKernel2DFn ScalarKn,
    TileKernelSpec2D CurrentKernel,
    typename... Args>
void tile2d(
    int64_t rows,
    int64_t cols,
    TileBorder border,
    ScalarKn&& scalar_impl,
    const CurrentKernel& current_kernel,
    const Args&... kernels
) {
    if constexpr (
        is_compiler_supported(CurrentKernel::INSTRUCTIONS)
        && std::is_same_v<scalar_t, typename CurrentKernel::TargetScalar>
    ) {
        if (is_supported(CurrentKernel::INSTRUCTIONS)) {
            return tile2d_autoblock<CurrentKernel::SIMD_BLOCK, scalar_t, ExecutionMode>(
                rows,
                cols,
                border,
                current_kernel.fn,
                std::forward<ScalarKn>(scalar_impl)
            );
        }
    }

    // Current kernel unusable (compiler can't emit it, or CPU lacks it):
    // drop it and try the remaining kernels with the same scalar fallback.
    return tile2d<scalar_t, ExecutionMode>(
        rows,
        cols,
        border,
        std::forward<ScalarKn>(scalar_impl),
        kernels...
    );
}

template<size_t SimdBlock, typename Scalar, TileKernel2DFn Fn>
constexpr TileKernel2D<SimdBlock, SimdSet::AVX2, Scalar, Fn> Avx2(Fn&& fn) {
    return TileKernel2D<SimdBlock, SimdSet::AVX2, Scalar, Fn>(fn);
}

template<size_t SimdBlock, typename Scalar, TileKernel2DFn Fn>
constexpr TileKernel2D<SimdBlock, SimdSet::AdvSIMD, Scalar, Fn> Neon(Fn&& fn) {
    return TileKernel2D<SimdBlock, SimdSet::AdvSIMD, Scalar, Fn>(fn);
}

template<size_t SimdBlock, typename Scalar, TileKernel2DFn Fn>
constexpr TileKernel2D<SimdBlock, SimdSet::WASM, Scalar, Fn> Wasm(Fn&& fn) {
    return TileKernel2D<SimdBlock, SimdSet::WASM, Scalar, Fn>(fn);
}

template<size_t SimdBlock, typename Scalar, TileKernel2DFn Fn>
constexpr TileKernel2D<SimdBlock, SimdSet::NONE, Scalar, Fn> Portable(Fn&& fn) {
    return TileKernel2D<SimdBlock, SimdSet::NONE, Scalar, Fn>(fn);
}

}  // namespace p10::simd

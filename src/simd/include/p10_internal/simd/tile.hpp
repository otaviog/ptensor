#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <utility>

#include <ptensor/tile_region2d.hpp>
#include "bitwise_math.hpp"
#include "cpuid.hpp"
#include "tile_execution.hpp"

namespace p10::simd {

template<typename F>
concept TileKernel2DFn = std::invocable<F, TileRegion2D>;

template<
    size_t CACHE_BLOCK,
    size_t SIMD_BLOCK,
    TileExecution ExecutionMode,
    TileKernel2DFn SimdFn,
    TileKernel2DFn ScalarFn>
void tile2d_blocked(int64_t rows, int64_t cols, SimdFn&& simd_impl, ScalarFn&& scalar_impl) {
    static_assert(CACHE_BLOCK % SIMD_BLOCK == 0, "CACHE_BLOCK must be a multiple of SIMD_BLOCK");

    constexpr int64_t CACHE = CACHE_BLOCK;
    constexpr int64_t SIMD = SIMD_BLOCK;
    const int64_t tile_rows = rows - bitwise_modulo<CACHE_BLOCK>(rows);
    const int64_t tile_cols = cols - bitwise_modulo<CACHE_BLOCK>(cols);

    // Main area [0, tile_rows) x [0, tile_cols), walked in cache blocks that are
    // further split into SIMD_BLOCK x SIMD_BLOCK tiles.
#pragma omp parallel for collapse(2) schedule(static) if (ExecutionMode == TileExecution::PARALLEL)
    for (int64_t block_row = 0; block_row < tile_rows; block_row += CACHE) {
        for (int64_t block_col = 0; block_col < tile_cols; block_col += CACHE) {
            for (int64_t simd_row = block_row; simd_row < block_row + CACHE; simd_row += SIMD) {
                for (int64_t simd_col = block_col; simd_col < block_col + CACHE; simd_col += SIMD) {
                    simd_impl(TileRegion2D {
                        .row = simd_row,
                        .col = simd_col,
                        .height = SIMD,
                        .width = SIMD
                    });
                }
            }
        }
    }

    // Right border: rows [0, tile_rows) x cols [tile_cols, cols).
    if (tile_cols < cols) {
        scalar_impl(TileRegion2D {
            .row = 0,
            .col = tile_cols,
            .height = tile_rows,
            .width = cols - tile_cols
        });
    }

    // Bottom border: rows [tile_rows, rows) x all cols.
    if (tile_rows < rows) {
        scalar_impl(
            TileRegion2D {.row = tile_rows, .col = 0, .height = rows - tile_rows, .width = cols}
        );
    }
}

// Round a target cache-tile side (in elements) down to a multiple of
// SIMD_BLOCK, never below one block. Guarantees tile2d_blocked's
// CACHE % SIMD_BLOCK == 0 invariant for any SIMD_BLOCK, so the divisibility
// static_assert can never fire from an auto-picked bucket.
template<size_t SIMD_BLOCK>
constexpr size_t snap_to_simd(size_t target) {
    const size_t blocks = target / SIMD_BLOCK;
    return (blocks == 0 ? size_t {1} : blocks) * SIMD_BLOCK;
}

template<size_t SIMD_BLOCK, typename scalar_t, TileKernel2DFn SimdKn, TileKernel2DFn ScalarKn>
void tile2d_autocache(int64_t rows, int64_t cols, SimdKn&& simd_impl, ScalarKn&& scalar_impl) {
    // Size the cache tile so its working set stays in L1d: a square side in
    // elements of ~sqrt(L1) bytes / element size (the src and dst tiles share L1).
    const int64_t processor_cache_size_sqrt =
        static_cast<int64_t>(std::sqrt(l1_cache_size())) / static_cast<int64_t>(sizeof(scalar_t));

    if (std::max(rows, cols) < processor_cache_size_sqrt) {
        std::forward<ScalarKn>(scalar_impl)(
            TileRegion2D {.row = 0, .col = 0, .height = rows, .width = cols}
        );
        return;
    }

    if (processor_cache_size_sqrt >= 1024) {
        tile2d_blocked<snap_to_simd<SIMD_BLOCK>(1024), SIMD_BLOCK, TileExecution::SEQUENTIAL>(
            rows, cols, std::forward<SimdKn>(simd_impl), std::forward<ScalarKn>(scalar_impl)
        );
    } else if (processor_cache_size_sqrt >= 512) {
        tile2d_blocked<snap_to_simd<SIMD_BLOCK>(512), SIMD_BLOCK, TileExecution::SEQUENTIAL>(
            rows, cols, std::forward<SimdKn>(simd_impl), std::forward<ScalarKn>(scalar_impl)
        );
    } else if (processor_cache_size_sqrt >= 256) {
        tile2d_blocked<snap_to_simd<SIMD_BLOCK>(256), SIMD_BLOCK, TileExecution::SEQUENTIAL>(
            rows, cols, std::forward<SimdKn>(simd_impl), std::forward<ScalarKn>(scalar_impl)
        );
    } else if (processor_cache_size_sqrt >= 128) {
        tile2d_blocked<snap_to_simd<SIMD_BLOCK>(128), SIMD_BLOCK, TileExecution::SEQUENTIAL>(
            rows, cols, std::forward<SimdKn>(simd_impl), std::forward<ScalarKn>(scalar_impl)
        );
    } else if (processor_cache_size_sqrt >= 64) {
        tile2d_blocked<snap_to_simd<SIMD_BLOCK>(64), SIMD_BLOCK, TileExecution::SEQUENTIAL>(
            rows, cols, std::forward<SimdKn>(simd_impl), std::forward<ScalarKn>(scalar_impl)
        );
    } else {
        tile2d_blocked<snap_to_simd<SIMD_BLOCK>(32), SIMD_BLOCK, TileExecution::SEQUENTIAL>(
            rows, cols, std::forward<SimdKn>(simd_impl), std::forward<ScalarKn>(scalar_impl)
        );
    }
}

template<size_t SimdBlock, SimdSet TargetInstructions, TileKernel2DFn KernelFn>
struct TileKernel2D {
    static constexpr size_t SIMD_BLOCK = SimdBlock;
    static constexpr SimdSet INSTRUCTIONS = TargetInstructions;
    KernelFn fn;
};

template<typename T>
concept TileKernel2DConcept = requires {
    { T::SIMD_BLOCK } -> std::convertible_to<size_t>;
    { T::INSTRUCTIONS } -> std::convertible_to<SimdSet>;
} && TileKernel2DFn<decltype(T::fn)>;

template<typename scalar_t, TileKernel2DFn ScalarKn>
void tile2d(
    int64_t rows,
    int64_t cols,
    ScalarKn&& scalar_impl) {
    scalar_impl(
        TileRegion2D {.row = 0, .col = 0, .height = rows, .width = cols});
}

template<typename scalar_t, TileKernel2DFn ScalarKn, TileKernel2DConcept CurrentKernel, typename... Args>
void tile2d(
    int64_t rows,
    int64_t cols,
    ScalarKn&& scalar_impl,
    const CurrentKernel& current_kern,
    const Args&... kernels
) {
    if constexpr (is_compiler_supported(CurrentKernel::INSTRUCTIONS)) {
        if (is_supported(CurrentKernel::INSTRUCTIONS)) {
            return tile2d_autocache<CurrentKernel::SIMD_BLOCK, scalar_t>(
                rows,
                cols,
                current_kern.fn,
                std::forward<ScalarKn>(scalar_impl)
            );
        }
    }

    // Current kernel unusable (compiler can't emit it, or CPU lacks it):
    // drop it and try the remaining kernels with the same scalar fallback.
    return tile2d<scalar_t>(
        rows,
        cols,
        std::forward<ScalarKn>(scalar_impl),
        kernels...
    );
}

template<size_t SimdBlock, TileKernel2DFn Fn>
constexpr TileKernel2D<SimdBlock, SimdSet::AVX2, Fn> Avx2(Fn &&fn) {
    return TileKernel2D<SimdBlock, SimdSet::AVX2, Fn>(fn);
}

template<size_t SimdBlock, TileKernel2DFn Fn>
constexpr TileKernel2D<SimdBlock, SimdSet::AdvSIMD, Fn> AdvSIMD(Fn &&fn) {
    return TileKernel2D<SimdBlock, SimdSet::AdvSIMD, Fn>(fn);
}

template<size_t SimdBlock, TileKernel2DFn Fn>
constexpr TileKernel2D<SimdBlock, SimdSet::WASM, Fn> Wasm(Fn &&fn) {
    return TileKernel2D<SimdBlock, SimdSet::WASM, Fn>(fn);
}

template<size_t SimdBlock, TileKernel2DFn Fn>
constexpr TileKernel2D<SimdBlock, SimdSet::NONE, Fn> GenericSimd(Fn &&fn) {
    return TileKernel2D<SimdBlock, SimdSet::NONE, Fn>(fn);
}

}  // namespace p10::simd

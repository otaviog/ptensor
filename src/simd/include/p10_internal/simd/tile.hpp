#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <utility>

#include <ptensor/tile_region2d.hpp>

#include "bitwise_math.hpp"
#include "cpuid.hpp"

namespace p10::simd {

struct TileRegion1D {
    int64_t offset;
    int64_t size;
};

template<typename F>
concept TileKernel1D = std::invocable<F, TileRegion1D>;

template<size_t CACHE_SIZE, size_t SIMD_SIZE, TileKernel1D SimdKn, TileKernel1D ScalarKn>
void tile1d(int64_t size, SimdKn&& simd_impl, ScalarKn&& scalar_impl) {
    static_assert(CACHE_SIZE % SIMD_SIZE == 0, "CACHE_SIZE must be a multiple of SIMD_SIZE");

    constexpr int64_t CACHE = CACHE_SIZE;
    constexpr int64_t SIMD = SIMD_SIZE;
    const int64_t tile_size = size - bitwise_modulo<SIMD_SIZE>(size);

    // Main: SIMD_SIZE chunks, grouped into CACHE_SIZE blocks for locality.
    for (int64_t block = 0; block < tile_size; block += CACHE) {
        const int64_t block_end = std::min(block + CACHE, tile_size);
        for (int64_t offset = block; offset < block_end; offset += SIMD) {
            simd_impl({.offset = offset, .size = SIMD});
        }
    }

    // Tail: leftover elements that don't fill a SIMD_SIZE chunk.
    if (tile_size < size) {
        scalar_impl(TileRegion1D {.offset = tile_size, .size = size - tile_size});
    }
}

template<size_t SIMD_SIZE, typename scalar_t, TileKernel1D SimdKn, TileKernel1D ScalarKn>
void dynamic_tile1d(int64_t size, SimdKn&& simd_impl, ScalarKn&& scalar_impl) {
    // Size the cache block so it stays in L1d (linear in 1D, hence L1 / element).
    const size_t cache_elems = l1_cache_size() / sizeof(scalar_t);

    if (size < static_cast<int64_t>(SIMD_SIZE)) {
        std::forward<ScalarKn>(scalar_impl)({.offset = 0, .size = size});
        return;
    }

    if (cache_elems >= 8192) {
        tile1d<8192, SIMD_SIZE>(
            size,
            std::forward<SimdKn>(simd_impl),
            std::forward<ScalarKn>(scalar_impl)
        );
    } else if (cache_elems >= 4096) {
        tile1d<4096, SIMD_SIZE>(
            size,
            std::forward<SimdKn>(simd_impl),
            std::forward<ScalarKn>(scalar_impl)
        );
    } else if (cache_elems >= 2048) {
        tile1d<2048, SIMD_SIZE>(
            size,
            std::forward<SimdKn>(simd_impl),
            std::forward<ScalarKn>(scalar_impl)
        );
    } else {
        tile1d<1024, SIMD_SIZE>(
            size,
            std::forward<SimdKn>(simd_impl),
            std::forward<ScalarKn>(scalar_impl)
        );
    }
}

template<typename F>
concept TileKernel2D = std::invocable<F, TileRegion2D>;

template<size_t CACHE_BLOCK, size_t SIMD_BLOCK, TileKernel2D SimdKn, TileKernel2D ScalarKn>
void tile2d(int64_t rows, int64_t cols, SimdKn&& simd_impl, ScalarKn&& scalar_impl) {
    static_assert(CACHE_BLOCK % SIMD_BLOCK == 0, "CACHE_BLOCK must be a multiple of SIMD_BLOCK");

    constexpr int64_t CACHE = CACHE_BLOCK;
    constexpr int64_t SIMD = SIMD_BLOCK;
    const int64_t tile_rows = rows - bitwise_modulo<CACHE_BLOCK>(rows);
    const int64_t tile_cols = cols - bitwise_modulo<CACHE_BLOCK>(cols);

    // Main area [0, tile_rows) x [0, tile_cols), walked in cache blocks that are
    // further split into SIMD_BLOCK x SIMD_BLOCK tiles.
#pragma omp parallel for collapse(2) schedule(static)
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

template<size_t SIMD_BLOCK, typename scalar_t, TileKernel2D SimdKn, TileKernel2D ScalarKn>
void dynamic_tile2d(int64_t rows, int64_t cols, SimdKn&& simd_impl, ScalarKn&& scalar_impl) {
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
        tile2d<1024, SIMD_BLOCK>(
            rows,
            cols,
            std::forward<SimdKn>(simd_impl),
            std::forward<ScalarKn>(scalar_impl)
        );
    } else if (processor_cache_size_sqrt >= 512) {
        tile2d<512, SIMD_BLOCK>(
            rows,
            cols,
            std::forward<SimdKn>(simd_impl),
            std::forward<ScalarKn>(scalar_impl)
        );
    } else if (processor_cache_size_sqrt >= 128) {
        tile2d<128, SIMD_BLOCK>(
            rows,
            cols,
            std::forward<SimdKn>(simd_impl),
            std::forward<ScalarKn>(scalar_impl)
        );
    } else if (processor_cache_size_sqrt >= 64) {
        tile2d<64, SIMD_BLOCK>(
            rows,
            cols,
            std::forward<SimdKn>(simd_impl),
            std::forward<ScalarKn>(scalar_impl)
        );
    } else {
        tile2d<32, SIMD_BLOCK>(
            rows,
            cols,
            std::forward<SimdKn>(simd_impl),
            std::forward<ScalarKn>(scalar_impl)
        );
    }
}

}  // namespace p10::simd

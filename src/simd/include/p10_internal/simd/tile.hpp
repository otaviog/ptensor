#pragma once

#include <concepts>

#include "../../../../core/include/span2d.hpp"
#include "bitwise_math.hpp"

namespace p10::simd {

struct TileRegion1D {
    ptrdiff_t offset;
    size_t size;
};

struct TileBorder1D {
    size_t left = 0;
    size_t right = 0;
};

template<typename F>
concept TileKernel1D = std::invocable<F, TileRegion1D>;

template<size_t CACHE_SIZE, size_t SIMD_SIZE, TileKernel1D GenericKn, TileKernel1D FallbackKn, TileBorder Border = TileBorder1D{}>
void tile1d(size_t size, GenericKn&& simd_impl, FallbackKn&& fallback_impl) {
    const size_t tile_size = size - bitwise_modulo<CACHE_SIZE>(size);

    for (size_t block = 0; block < tile_size; block += CACHE_SIZE) {
        for (size_t simd = block; simd < block + CACHE_SIZE; simd += SIMD_SIZE) {
            simd_impl({static_cast<ptrdiff_t>(simd), SIMD_SIZE});
        }
    }

    fallback_impl({static_cast<ptrdiff_t>(CACHE_SIZE), CACHE_SIZE - size});
}

template<size_t SIMD_SIZE, TileKernel1D GenericKn, TileKernel1D ScalarKn>
void dynamic_tile1d(size_t size, GenericKn simd_impl, ScalarKn &&scalar_impl) {
    // Todo
}


template<typename F>
concept TileKernel2D = std::invocable<F, TileRegion2D>;

template<size_t CACHE_BLOCK, size_t SIMD_BLOCK, typename Generic, typename Fallback>
void tile2d(size_t rows, size_t cols, Generic&& generic_impl, Fallback&& fallback_impl) {
    const size_t tile_rows = rows - bitwise_modulo<CACHE_BLOCK>(rows);
    const size_t tile_cols = cols - bitwise_modulo<CACHE_BLOCK>(cols);

    for (size_t block_row = 0; block_row < tile_rows; block_row += CACHE_BLOCK) {
        for (size_t block_col = 0; block_col < tile_cols; block_col += CACHE_BLOCK) {
            for (size_t simd_row = block_row; simd_row < block_row + CACHE_BLOCK;
                 simd_row += SIMD_BLOCK) {
                for (size_t simd_col = block_col; simd_col < block_col + CACHE_BLOCK;
                     simd_col += SIMD_BLOCK) {
                    simd_impl(SpanRegion2D {
                        simd_row,
                        simd_col,
                    });
                }
            }
        }
    }

    const SpanRegion2D right_region {
        tile_cols,
        tile_rows,  // We process all the bottom cols, so dont bother to do it in the right
        cols - tile_cols,
        tile_cols
    };
    fallback_impl(right_region);

    const SpanRegion2D bottom_region {tile_rows * tile_cols, rows - tile_rows, cols, rows};
    fallback_impl(bottom_region);
}

template<size_t SIMD_BLOCK, typename Generic, typename Fallback>
bool dynamic_tile2d(size_t rows, size_t cols, Generic&& generic_impl, Fallback&& fallback_impl) {
    const size_t processor_cache_size = 1024 * 1024 * 5;  // TODO: grab the actual cache size

    if (processor_cache_size > 1024) {
        tile<64, SIMD_BLOCK>(rows, cols, generic_impl, fallback_impl);
    } else if (processor_cache_size > 512) {
        tile<32, SIMD_BLOCK>(rows, cols, generic_impl, fallback_impl);
    } else {
        return false;
    }

    return true;
}

}  // namespace p10::simd

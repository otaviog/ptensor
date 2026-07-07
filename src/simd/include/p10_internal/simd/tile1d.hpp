#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <utility>

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

}  // namespace p10::simd

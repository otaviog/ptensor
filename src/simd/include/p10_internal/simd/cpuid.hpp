#pragma once

#include <cstddef>
#include <cinttypes>

namespace p10::simd {
enum class SimdSet: uint8_t {
    AVX2 = 0, WASM = 1, AdvSIMD = 2
};

bool is_supported(SimdSet set);

size_t l1_cache_size();
size_t l2_cache_size();
size_t l3_cache_size();

}  // namespace p10::simd

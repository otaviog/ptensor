#pragma once

#include <cstddef>
#include <cinttypes>

namespace p10::simd {
enum class SimdSet: uint8_t {
    NONE = 0, AVX2 = 1, WASM = 2, AdvSIMD = 3
};

bool is_supported(SimdSet set);

size_t l1_cache_size();
size_t l2_cache_size();
size_t l3_cache_size();

}  // namespace p10::simd

#if defined(__clang__)
#include "cpuid.is_compiler_supported.clang.hpp"
#endif

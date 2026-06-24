#pragma once

#include "cpuid.hpp"

namespace p10::simd {

constexpr bool is_compiler_supported(SimdSet set) {
#if defined(_M_X64) || defined(_M_IX86)
    return set == SimdSet::AVX2 || set == SimdSet::NONE;
#endif

#if defined(_M_ARM64)
    return set == SimdSet::AdvSIMD || set == SimdSet::NONE;
#endif

    return set == SimdSet::NONE;
}

}  // namespace p10::simd

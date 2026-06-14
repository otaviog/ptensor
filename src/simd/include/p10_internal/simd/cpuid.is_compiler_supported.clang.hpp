#pragma once

#include "cpuid.hpp"

namespace p10::simd {

constexpr bool is_compiler_supported(SimdSet set) {
#if defined(__x86_64__) || defined(__i386__)
    return set == SimdSet::AVX2 || set == SimdSet::NONE;
#endif

#ifdef __wasm_simd128__
    return set == SimdSet::WASM || set == SimdSet::NONE;
#endif

#if defined(__aarch64__)
    return set == SimdSet::AdvSIMD || set == SimdSet::NONE;
#endif
    
    return set == SimdSet::NONE;
}

}

#pragma once

#include "cpuid.hpp"

namespace p10::simd {

constexpr bool is_compiler_supported(SimdSet set) {
#if defined(__x86_64__) || defined(__i386__)
    return set == SimdSet::AVX2;
#endif

#ifdef __wasm_simd128__
    return set == SimdSet::WASM;
#endif

#if defined(__aarch64__)
    return set == SimdSet::AdvSIMD;
#endif
    
    return set == SimdSet::NONE;
}

}

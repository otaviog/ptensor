#pragma once

#ifdef _MSC_VER
    #include <intrin.h>
#endif

#if defined(_MSC_VER) || \
    ((defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)) && \
     (__has_include(<intrinsics.h>) || __has_include(<immintrin.h>)))
    #define PTENSOR_HAS_INTRINSICS_H 1
#endif

namespace p10::simd {
inline bool is_avx2_supported() {
#ifdef _MSC_VER
    int cpuInfo[4];
    __cpuid(cpuInfo, 0);
    int nIds = cpuInfo[0];

    if (nIds >= 7) {
        __cpuidex(cpuInfo, 7, 0);
        return (cpuInfo[1] & (1 << 5)) != 0;  // EBX bit 5 is AVX2
    }
    return false;
#elif defined(__x86_64__) || defined(__i386__)
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}
}  // namespace p10::simd
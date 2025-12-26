#pragma once

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace p10::simd {
inline bool is_avx2_supported() {
#if defined(_MSC_VER)
    int cpuInfo[4];
    __cpuid(cpuInfo, 0);
    int nIds = cpuInfo[0];

    if (nIds >= 7) {
        __cpuidex(cpuInfo, 7, 0);
        return (cpuInfo[1] & (1 << 5)) != 0;  // EBX bit 5 is AVX2
    }
    return false;
#else
    return __builtin_cpu_supports("avx2");
#endif
}
}  // namespace p10::simd
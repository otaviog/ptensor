#pragma once

#if defined(_MSC_VER)
    #include <intrin.h>
#endif

#if __has_include(<intrinsics.h>)
    #define PTENSOR_HAS_INTRINSICS_H
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
#elif defined(__x86_64__) || defined(__i386__)
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}
}  // namespace p10::simd
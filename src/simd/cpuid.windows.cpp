#include <intrin.h>
#include <windows.h>

#include <p10_internal/simd/cpuid.hpp>

namespace p10::simd {

namespace {
bool detect_avx2() {
    int cpu_info[4] = {};
    __cpuid(cpu_info, 0);
    if (cpu_info[0] < 7) { return false; }
    __cpuidex(cpu_info, 7, 0);
    return (cpu_info[1] & (1 << 5)) != 0;  // EBX bit 5 = AVX2
}

bool detect_adv_simd() {
#if defined(_M_ARM64)
    return IsProcessorFeaturePresent(PF_ARM_NEON_INSTRUCTIONS_AVAILABLE) != FALSE;
#else
    return false;
#endif
}

size_t read_cache_size(int level) {
    DWORD buffer_size = 0;
    GetLogicalProcessorInformation(nullptr, &buffer_size);
    if (buffer_size == 0) { return 0; }

    const auto count = buffer_size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
    auto* buffer = new SYSTEM_LOGICAL_PROCESSOR_INFORMATION[count];
    if (GetLogicalProcessorInformation(buffer, &buffer_size) == FALSE) {
        delete[] buffer;
        return 0;
    }

    size_t size = 0;
    for (DWORD i = 0; i < count; i++) {
        if (buffer[i].Relationship != RelationCache) { continue; }
        const auto& cache = buffer[i].Cache;
        if (cache.Level == static_cast<BYTE>(level) &&
            cache.Type != CacheInstruction) {
            size = cache.Size;
            break;
        }
    }
    delete[] buffer;
    return size;
}

size_t detect_l1() { return read_cache_size(1); }
size_t detect_l2() { return read_cache_size(2); }
size_t detect_l3() { return read_cache_size(3); }
}  // namespace

bool is_supported(SimdSet set) {
    static const bool AVX2 = detect_avx2();
    static const bool ADV_SIMD = detect_adv_simd();
    switch (set) {
    case SimdSet::AVX2: return AVX2;
    case SimdSet::WASM: return false;
    case SimdSet::AdvSIMD: return ADV_SIMD;
    case SimdSet::NONE: return true;
    }
    return false;
}

size_t l1_cache_size() {
    static const size_t L1 = detect_l1();
    return L1;
}

size_t l2_cache_size() {
    static const size_t L2 = detect_l2();
    return L2;
}

size_t l3_cache_size() {
    static const size_t L3 = detect_l3();
    return L3;
}

}  // namespace p10::simd

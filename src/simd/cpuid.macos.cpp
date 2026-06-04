#if defined(__x86_64__) || defined(__i386__)
    #include <cpuid.h>
#endif

#include <sys/sysctl.h>

#include <p10_internal/simd/cpuid.hpp>

namespace p10::simd {

namespace {
size_t sysctl_size(const char* name) {
    size_t value = 0;
    size_t len = sizeof(value);
    sysctlbyname(name, &value, &len, nullptr, 0);
    return value;
}

bool sysctl_bool(const char* name) {
    int value = 0;
    size_t len = sizeof(value);
    sysctlbyname(name, &value, &len, nullptr, 0);
    return value != 0;
}
}  // namespace

bool is_supported(SimdSet set) {
    static const bool AVX2 =
#if defined(__x86_64__) || defined(__i386__)
        __builtin_cpu_supports("avx2");
#else
        false;
#endif
    static const bool ADV_SIMD =
#if defined(__aarch64__)
        sysctl_bool("hw.optional.neon");
#else
        false;
#endif
    switch (set) {
    case SimdSet::AVX2: return AVX2;
    case SimdSet::WASM: return false;
    case SimdSet::AdvSIMD: return ADV_SIMD;
    }
    return false;
}

size_t l1_cache_size() {
    static const size_t L1 = sysctl_size("hw.l1dcachesize");
    return L1;
}

size_t l2_cache_size() {
    static const size_t L2 = sysctl_size("hw.l2cachesize");
    return L2;
}

size_t l3_cache_size() {
    static const size_t L3 = sysctl_size("hw.l3cachesize");
    return L3;
}

}  // namespace p10::simd

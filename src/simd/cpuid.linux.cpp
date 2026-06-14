#include <cstdio>
#include <cstdlib>

#if defined(__aarch64__) && defined(__linux__)
    #include <asm/hwcap.h>
    #include <sys/auxv.h>
#endif

#include <p10_internal/simd/cpuid.hpp>

namespace p10::simd {

bool is_supported(SimdSet set) {
    static const bool AVX2 =
#if defined(__x86_64__) || defined(__i386__)
        __builtin_cpu_supports("avx2");
#else
        false;
#endif
    static const bool ADV_SIMD =
#if defined(__aarch64__) && defined(__linux__)
        (getauxval(AT_HWCAP) & HWCAP_ASIMD) != 0;
#else
        false;
#endif
    switch (set) {
    case SimdSet::AVX2: return AVX2;
    case SimdSet::WASM: return false;
    case SimdSet::AdvSIMD: return ADV_SIMD;
    case SimdSet::NONE: return true;
    }
    return false;
}

namespace {

size_t read_cache_size(int level) {
    char path[128];
    for (int i = 0; i < 8; i++) {
        (void)snprintf(path, sizeof(path),
                       "/sys/devices/system/cpu/cpu0/cache/index%d/level", i);
        FILE* f = fopen(path, "r");
        if (f == nullptr) { break; }
        char buf[32] = {};
        (void)fscanf(f, "%31s", buf);
        (void)fclose(f);
        if (strtol(buf, nullptr, 10) != level) { continue; }

        (void)snprintf(path, sizeof(path),
                       "/sys/devices/system/cpu/cpu0/cache/index%d/type", i);
        f = fopen(path, "r");
        if (f == nullptr) { continue; }
        char type[32] = {};
        (void)fscanf(f, "%31s", type);
        (void)fclose(f);
        if (type[0] == 'I') { continue; }  // skip Instruction-only cache

        (void)snprintf(path, sizeof(path),
                       "/sys/devices/system/cpu/cpu0/cache/index%d/size", i);
        f = fopen(path, "r");
        if (f == nullptr) { continue; }
        char size_buf[32] = {};
        (void)fscanf(f, "%31s", size_buf);
        (void)fclose(f);

        char* end = nullptr;
        auto size = static_cast<size_t>(strtoul(size_buf, &end, 10));
        if (end != nullptr && *end == 'M') { return size * 1024 * 1024; }
        return size * 1024;  // default unit is K
    }
    return 0;
}

size_t detect_l1() { return read_cache_size(1); }
size_t detect_l2() { return read_cache_size(2); }
size_t detect_l3() { return read_cache_size(3); }
}  // namespace

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


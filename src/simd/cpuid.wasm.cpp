#include <p10_internal/simd/cpuid.hpp>

namespace p10::simd {

bool is_supported(SimdSet set) {
#ifdef __wasm_simd128__
    if (set == SimdSet::WASM) { return true; }
#endif
    
    return set == SimdSet::NONE;
}

// Conservative defaults representative of typical hardware running WASM engines.
size_t l1_cache_size() { return 32 * 1024; }        // 32 KB
size_t l2_cache_size() { return 256 * 1024; }       // 256 KB
size_t l3_cache_size() { return 4 * 1024 * 1024; }  // 4 MB

}  // namespace p10::simd

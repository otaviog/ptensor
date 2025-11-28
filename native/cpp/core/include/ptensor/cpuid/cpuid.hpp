#pragma once

/// This file provides simple wrapper to CPUID and is NOT part of the public API. It should be moved to a new intern library. Put here so other modules can use it.
namespace p10 {
inline bool is_avx2_supported() {
    return __builtin_cpu_supports("avx2");
}
}  // namespace p10
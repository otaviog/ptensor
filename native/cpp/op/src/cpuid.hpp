#pragma once

namespace p10::op {
inline bool is_avx2_supported() {
    return __builtin_cpu_supports("avx2");
}
}  // namespace p10::op
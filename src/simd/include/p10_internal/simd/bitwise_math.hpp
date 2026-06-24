#pragma once
#include <cstddef>
#include <cstdint>

namespace p10::simd {
template<size_t b>
constexpr int64_t bitwise_modulo(int64_t a) {
    static_assert((b & (b - 1)) == 0, "b must be a power of two");
    return a & static_cast<int64_t>(b - 1);
}

}  // namespace p10::simd

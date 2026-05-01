#pragma once
#include <cstddef>

namespace p10::simd {
template<size_t b>
constexpr inline size_t bitwise_modulo(size_t a) {
    static_assert((b & (b - 1)) == 0, "b must be a power of two");
    return a & (b - 1);
}

}  // namespace p10::simd
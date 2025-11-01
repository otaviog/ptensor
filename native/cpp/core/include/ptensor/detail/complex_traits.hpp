#pragma once

#include <complex>

#include <type_traits>

namespace p10 {
namespace detail {
    template<typename T>
    struct is_complex: std::false_type {};

    template<typename T>
    struct is_complex<std::complex<T>>: std::true_type {};

    template<typename T>
    inline constexpr bool is_complex_v = is_complex<T>::value;
}  // namespace detail
}  // namespace p10
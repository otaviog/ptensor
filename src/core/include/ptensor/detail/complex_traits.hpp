#pragma once

#include <complex>

#include <type_traits>

namespace p10::detail {
template<typename T>
struct is_complex: std::false_type {};

template<typename T>
struct is_complex<std::complex<T>>: std::true_type {};

template<typename T>
inline constexpr bool IS_COMPLEX_V = is_complex<T>::value;
}  // namespace p10::detail

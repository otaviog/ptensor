#pragma once

#include <cinttypes>
#include <cstdint>
#include <limits>

#include <type_traits>

namespace p10::op::detail {

template<typename T>
struct has_fixed_point_type: public std::false_type {};

template<>
struct has_fixed_point_type<int8_t>: public std::true_type {};

template<>
struct has_fixed_point_type<int16_t>: public std::true_type {};

template<>
struct has_fixed_point_type<int32_t>: public std::true_type {};

template<>
struct has_fixed_point_type<uint8_t>: public std::true_type {};

template<>
struct has_fixed_point_type<uint16_t>: public std::true_type {};

template<>
struct has_fixed_point_type<uint32_t>: public std::true_type {};

template<>
struct has_fixed_point_type<float>: public std::true_type {};

template<>
struct has_fixed_point_type<double>: public std::true_type {};

template<typename T>
struct fixed_point_type {
    using type = T;
    static constexpr T factor = T{};
};

template<>
struct fixed_point_type<uint8_t> {
    using type = int16_t;
    static constexpr int16_t factor = 256;  // Full range of uint8_t
};

template<>
struct fixed_point_type<int8_t> {
    using type = int16_t;
    static constexpr int16_t factor = 256;  // Full range of int8_t
};

template<>
struct fixed_point_type<uint16_t> {
    using type = int32_t;
    static constexpr int32_t factor = 65536;  // Full range of uint16_t
};

template<>
struct fixed_point_type<int16_t> {
    using type = int32_t;
    static constexpr int32_t factor = 65536;  // Full range of int16_t
};

template<>
struct fixed_point_type<uint32_t> {
    using type = int64_t;
    static constexpr int64_t factor = (1LL << 32);  // Full range of uint32_t
};

template<>
struct fixed_point_type<int32_t> {
    using type = int64_t;
    static constexpr int64_t factor = (1LL << 32);  // Full range of int32_t
};

template<>
struct fixed_point_type<float> {
    using type = float;
    static constexpr float factor = 1.0f;
};

template<>
struct fixed_point_type<double> {
    using type = double;
    static constexpr double factor = 1.0;
};

}  // namespace p10::op::detail

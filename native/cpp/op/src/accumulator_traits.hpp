#pragma once

#include <cinttypes>
#include <cstdint>
#include <limits>

#include <type_traits>

namespace p10::op::detail {

template<typename T>
struct has_accumulator_traits: public std::false_type {};

template<>
struct has_accumulator_traits<int8_t>: public std::true_type {};

template<>
struct has_accumulator_traits<int16_t>: public std::true_type {};

template<>
struct has_accumulator_traits<int32_t>: public std::true_type {};

template<>
struct has_accumulator_traits<uint8_t>: public std::true_type {};

template<>
struct has_accumulator_traits<uint16_t>: public std::true_type {};

template<>
struct has_accumulator_traits<uint32_t>: public std::true_type {};

template<>
struct has_accumulator_traits<float>: public std::true_type {};

template<>
struct has_accumulator_traits<double>: public std::true_type {};

struct AccumulatorNotDefined {};

template<typename T>
struct accumulator_traits {
    using scalar_type = T;
    using accum_type = AccumulatorNotDefined;

    static constexpr accum_type from_float(float) {
        static_assert(
            sizeof(T) == 0,
            "accumulator_traits not defined for this type"
        );
        return {};
    }

    static constexpr scalar_type to_scalar(accum_type) {
        static_assert(
            sizeof(T) == 0,
            "accumulator_traits not defined for this type"
        );
        return {};
    }
};

template<>
struct accumulator_traits<uint8_t> {
    using scalar_type = uint8_t;
    using accum_type = int16_t;

    static constexpr accum_type from_float(float value) {
        return static_cast<accum_type>(value * 256.0f);  // 2^8
    }

    static constexpr scalar_type to_scalar(accum_type value) {
        return scalar_type(value >> 8);  // Divide by 256
    }
};

template<>
struct accumulator_traits<int8_t> {
    using scalar_type = int8_t;
    using accum_type = int16_t;

    static constexpr accum_type from_float(float value) {
        return static_cast<accum_type>(value * 256.0f);  // 2^8
    }

    static constexpr scalar_type to_scalar(accum_type value) {
        return scalar_type(value >> 8);  // Divide by 256
    }
};

template<>
struct accumulator_traits<uint16_t> {
    using scalar_type = uint16_t;
    using accum_type = int32_t;

    static constexpr accum_type from_float(float value) {
        return static_cast<accum_type>(value * 65536.0f);  // 2^16
    }

    static constexpr scalar_type to_scalar(accum_type value) {
        return scalar_type(value >> 16);  // Divide by 65536
    }
};

template<>
struct accumulator_traits<int16_t> {
    using scalar_type = int16_t;
    using accum_type = int32_t;

    static constexpr accum_type from_float(float value) {
        return static_cast<accum_type>(value * 65536.0f);  // 2^16
    }

    static constexpr scalar_type to_scalar(accum_type value) {
        return scalar_type(value >> 16);  // Divide by 65536
    }
};

template<>
struct accumulator_traits<uint32_t> {
    using scalar_type = uint32_t;
    using accum_type = int64_t;

    static constexpr accum_type from_float(float value) {
        return static_cast<accum_type>(value * 4294967296.0);  // 2^32
    }

    static constexpr scalar_type to_scalar(accum_type value) {
        return scalar_type(value >> 32);  // Divide by 2^32
    }
};

template<>
struct accumulator_traits<int32_t> {
    using scalar_type = int32_t;
    using accum_type = int64_t;

    static constexpr accum_type from_float(float value) {
        return static_cast<accum_type>(value * 4294967296.0);  // 2^32
    }

    static constexpr scalar_type to_scalar(accum_type value) {
        return scalar_type(value >> 32);  // Divide by 2^32
    }
};

template<>
struct accumulator_traits<float> {
    using scalar_type = float;
    using accum_type = float;

    static constexpr accum_type  from_float(float value) {
        return value;
    }

    static constexpr scalar_type to_scalar(accum_type value) {
        return value;
    }
};

template<>
struct accumulator_traits<double> {
    using scalar_type = double;
    using accum_type = double;

    static constexpr accum_type  from_float(float value) {
        return value;
    }

    static constexpr scalar_type to_scalar(accum_type value) {
        return value;
    }
};

}  // namespace p10::op::detail

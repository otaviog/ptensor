#pragma once
#include <cassert>
#include <cinttypes>
#include <span>
#include <string>

#include <ptensor/ptensor_dtype.h>

#include "detail/panic.hpp"

namespace p10 {

struct Dtype {
    enum Code : uint8_t {
        Float32 = P10_DTYPE_FLOAT32,
        Float64 = P10_DTYPE_FLOAT64,
        Float16 = P10_DTYPE_FLOAT16,
        Uint8 = P10_DTYPE_UINT8,
        Uint16 = P10_DTYPE_UINT16,
        Uint32 = P10_DTYPE_UINT32,
        Int8 = P10_DTYPE_INT8,
        Int16 = P10_DTYPE_INT16,
        Int32 = P10_DTYPE_INT32,
        Int64 = P10_DTYPE_INT64
    };

    template<typename T>
    static Dtype from() {
        using ActualType = std::remove_cv_t<std::remove_reference_t<T>>;
        // clang-format off
        if constexpr (std::is_same_v<ActualType, float>) return Dtype(Float32);
        else if constexpr (std::is_same_v<ActualType, double>) return Dtype(Float64);
        else if constexpr (std::is_same_v<ActualType, uint8_t>) return Dtype(Uint8);
        else if constexpr (std::is_same_v<ActualType, uint16_t>) return Dtype(Uint16);
        else if constexpr (std::is_same_v<ActualType, uint32_t>) return Dtype(Uint32);
        else if constexpr (std::is_same_v<ActualType, int8_t>) return Dtype(Int8);
        else if constexpr (std::is_same_v<ActualType, int16_t>) return Dtype(Int16);
        else if constexpr (std::is_same_v<ActualType, int32_t>) return Dtype(Int32);
        else if constexpr (std::is_same_v<ActualType, int64_t>) return Dtype(Int64);
        else static_assert(!std::is_same_v<ActualType, ActualType>, "Unsupported type for Dtype::from<T>()");
        // clang-format on
    }

    Dtype() = default;

    constexpr Dtype(Code value) : value(value) {}

    constexpr operator Code() const {
        return value;
    }

    explicit operator bool() const = delete;

    std::size_t size() const {
        switch (value) {
            case Uint8:
            case Int8:
                return 1;
            case Float16:
            case Uint16:
            case Int16:
                return 2;
            case Uint32:
            case Int32:
            case Float32:
                return 4;
            case Int64:
            case Float64:
                return 8;
            default:
                return 0;
        }
    }

    template<typename F>
    auto visit(F&& visitor, std::span<std::byte> data) const {
        return do_visit(std::forward<F>(visitor), data);
    }

    template<typename F>
    auto visit(F&& visitor, std::span<const std::byte> data) const {
        return do_visit(std::forward<F>(visitor), data);
    }

    Code value = Dtype::Float32;

  private:
    template<typename F, typename ByteType>
    auto do_visit(F&& visitor, std::span<ByteType> data) const {
        switch (value) {
            case Uint8:
                return do_type_visit<F, uint8_t, ByteType>(std::forward<F>(visitor), data);
                break;
            case Uint16:
                return do_type_visit<F, uint16_t, ByteType>(std::forward<F>(visitor), data);
                break;
            case Uint32:
                return do_type_visit<F, uint32_t, ByteType>(std::forward<F>(visitor), data);
                break;
            case Int8:
                return do_type_visit<F, int8_t, ByteType>(std::forward<F>(visitor), data);
                break;
            case Int16:
                return do_type_visit<F, int16_t, ByteType>(std::forward<F>(visitor), data);
                break;
            case Int32:
                return do_type_visit<F, int32_t, ByteType>(std::forward<F>(visitor), data);
                break;
            case Int64:
                return do_type_visit<F, int64_t, ByteType>(std::forward<F>(visitor), data);
                break;
            case Float32:
                return do_type_visit<F, float, ByteType>(std::forward<F>(visitor), data);
                break;
            case Float64:
                return do_type_visit<F, double, ByteType>(std::forward<F>(visitor), data);
                break;
            default:
                detail::panic("Unsupported dtype in Dtype::visit()");
        }
    }

    template<typename F, typename T, typename ByteType>
    auto do_type_visit(F&& visitor, std::span<ByteType> data) const {
        if constexpr (std::is_const<ByteType>::value) {
            return visitor(
                std::span(reinterpret_cast<const T*>(data.data()), data.size() / sizeof(T))
            );
        } else {
            return visitor(std::span(reinterpret_cast<T*>(data.data()), data.size() / sizeof(T)));
        }
    }
};

std::string to_string(Dtype dtype);

}  // namespace p10

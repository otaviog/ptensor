#pragma once

#include <ptensor/p10_result.hpp>

#include "common.hpp"

template<typename T>
inline T js_unwrap(p10::P10Result<T>&& value) {
    if (value.is_error()) {
        throw value.err();
    }
    return value.unwrap();
}

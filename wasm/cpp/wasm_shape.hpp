#pragma once

#include <emscripten/val.h>
#include <ptensor/shape.hpp>
#include "ptensor/p10_error.hpp"

using namespace emscripten;

class WasmShape : public p10::Shape {
  public:
    WasmShape() = default;

    static WasmShape fromArray(const val& dims_array) {
        const auto length = dims_array["length"].as<size_t>();
        p10::Shape shape = p10::Shape::zeros(length).unwrap();
        auto shape_s = shape.as_span();
        for (size_t i = 0; i < length; ++i) {
            shape_s[i] = dims_array[i].as<int64_t>();
        }

        

        return WasmShape(shape);
    }

    val toArray() const {
        val arr = val::array();
        const auto span = as_span();
        for (size_t i = 0; i < span.size(); ++i) {
            arr.set(i, span[i]);
        }
        return arr;
    }

    WasmShape(const p10::Shape& shape) : p10::Shape(shape) {}

    std::string toString() const {
        return p10::to_string(*this);
    }
};

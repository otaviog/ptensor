#pragma once

#include <emscripten/val.h>
#include <ptensor/shape.hpp>

#include "common.hpp"
#include "js_error.hpp"

class JsShape: public p10::Shape {
  public:
    JsShape() = default;

    static JsShape fromArray(const val& dims_array) {
        const auto length = dims_array["length"].as<size_t>();
        p10::Shape shape = js_unwrap(p10::Shape::zeros(length));
        auto shape_s = shape.as_span();
        for (size_t i = 0; i < length; ++i) {
            shape_s[i] = dims_array[i].as<int64_t>();
        }

        return JsShape(shape);
    }

    val toArray() const {
        val arr = val::array();
        const auto span = as_span();
        for (size_t i = 0; i < span.size(); ++i) {
            arr.set(i, span[i]);
        }
        return arr;
    }

    JsShape(const p10::Shape& shape) : p10::Shape(shape) {}

    std::string toString() const {
        return p10::to_string(*this);
    }
};

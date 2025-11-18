#include <emscripten/bind.h>
#include <ptensor/dtype.hpp>
#include <ptensor/p10_error.hpp>
#include <ptensor/shape.hpp>

#include "js_shape.hpp"
#include "js_tensor.hpp"

using namespace emscripten;

EMSCRIPTEN_BINDINGS(P10) {
    class_<p10::P10Error>("P10Error")
        .function("code", &p10::P10Error::code)
        .function("toString", &p10::P10Error::to_string)
        .function("isOk", &p10::P10Error::is_ok)
        .function("isError", &p10::P10Error::is_error);

    class_<JsShape>("Shape")
        .constructor<>()
        .class_function("fromArray", &JsShape::fromArray)
        .function("toArray", &JsShape::toArray)
        .function("dims", &JsShape::dims)
        .function("count", &JsShape::count)
        .function("empty", &JsShape::empty)
        .function("toString", &JsShape::toString);

    class_<p10::Dtype>("Dtype")
        .constructor<>()
        .constructor<p10::Dtype::Code>()
        .class_function("fromString", optional_override([](const std::string& type_str) -> p10::Dtype {
            auto result = p10::Dtype::from(type_str);
            if (result.is_error()) {
                throw std::runtime_error(result.err().to_string());
            }
            return result.unwrap();
        }))
        .function("toString", optional_override([](const p10::Dtype& dtype) -> std::string {
            return p10::to_string(dtype);
        }));

    enum_<p10::Dtype::Code>("DtypeCode")
        .value("Float32", p10::Dtype::Float32)
        .value("Float64", p10::Dtype::Float64)
        .value("Float16", p10::Dtype::Float16)
        .value("Uint8", p10::Dtype::Uint8)
        .value("Uint16", p10::Dtype::Uint16)
        .value("Uint32", p10::Dtype::Uint32)
        .value("Int8", p10::Dtype::Int8)
        .value("Int16", p10::Dtype::Int16)
        .value("Int32", p10::Dtype::Int32)
        .value("Int64", p10::Dtype::Int64);

    class_<JsTensor>("Tensor")
        .constructor<>()
        .class_function("zeros", &JsTensor::zeros, allow_raw_pointers())
        .class_function("fromData", &JsTensor::fromData, allow_raw_pointers())
        .function("getSize", &JsTensor::getSize)
        .function("getShape", &JsTensor::getShape)
        .function("getDtype", &JsTensor::getDtype)
        .function("toString", &JsTensor::toString);
}

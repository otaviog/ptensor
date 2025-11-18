#include <emscripten/bind.h>

#include <ptensor/dtype.hpp>
#include <ptensor/p10_error.hpp>
#include <ptensor/shape.hpp>

#include "wasm_shape.hpp"
#include "wasm_tensor.hpp"

using namespace emscripten;

// Helper function to create Dtype from string
p10::Dtype createDtypeFromString(const std::string& dtype_str) {
    auto result = p10::Dtype::from(dtype_str);
    if (result.is_error()) {
        throw std::runtime_error(result.err().to_string());
    }
    return result.unwrap();
}

EMSCRIPTEN_BINDINGS(P10) {
    class_<p10::P10Error>("P10Error")
        .function("code", &p10::P10Error::code)
        .function("toString", &p10::P10Error::to_string)
        .function("isOk", &p10::P10Error::is_ok)
        .function("isError", &p10::P10Error::is_error);

    class_<WasmShape>("Shape")
        .constructor<>()
        .class_function("fromArray", &WasmShape::fromArray)
        .function("toArray", &WasmShape::toArray)
        .function("dims", &WasmShape::dims)
        .function("count", &WasmShape::count)
        .function("empty", &WasmShape::empty)
        .function("toString", &WasmShape::toString);

    class_<p10::Dtype>("Dtype")
        .constructor<>()
        .constructor<p10::Dtype::Code>()
        .function("toString", &p10::Dtype::to_string);

    function("createDtypeFromString", &createDtypeFromString);

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

    class_<WasmTensor>("Tensor")
        .constructor<>()
        .class_function("zeros", &WasmTensor::zeros, allow_raw_pointers())
        .function("fromData", &WasmTensor::fromData)
        .function("getSize", &WasmTensor::getSize)
        .function("getShape", &WasmTensor::getShape)
        .function("getDtypeStr", &WasmTensor::getDtype)
        .function("toString", &WasmTensor::toString);
}

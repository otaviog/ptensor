#include "dtype.hpp"

#include "p10_result.hpp"

namespace p10 {
P10Result<Dtype> Dtype::from(const std::string& type_str) {
    if (type_str == "uint8") {
        return Ok<Dtype>(Dtype::Uint8);
    } else if (type_str == "uint16") {
        return Ok<Dtype>(Dtype::Uint16);
    } else if (type_str == "uint32") {
        return Ok<Dtype>(Dtype::Uint32);
    } else if (type_str == "int8") {
        return Ok<Dtype>(Dtype::Int8);
    } else if (type_str == "int16") {
        return Ok<Dtype>(Dtype::Int16);
    } else if (type_str == "int32") {
        return Ok<Dtype>(Dtype::Int32);
    } else if (type_str == "int64") {
        return Ok<Dtype>(Dtype::Int64);
    } else if (type_str == "float16") {
        return Ok<Dtype>(Dtype::Float16);
    } else if (type_str == "float32") {
        return Ok<Dtype>(Dtype::Float32);
    } else if (type_str == "float64") {
        return Ok<Dtype>(Dtype::Float64);
    } else {
        return Err(P10Error(P10Error::InvalidArgument, "Unknown dtype string: " + type_str));
    }
}

const char* to_cstring(Dtype::Code dtype) {
    switch (dtype) {
        case Dtype::Uint8:
            return "uint8";
        case Dtype::Uint16:
            return "uint16";
        case Dtype::Uint32:
            return "uint32";
        case Dtype::Int8:
            return "int8";
        case Dtype::Int16:
            return "int16";
        case Dtype::Int32:
            return "int32";
        case Dtype::Int64:
            return "int64";
        case Dtype::Float16:
            return "float16";
        case Dtype::Float32:
            return "float32";
        case Dtype::Float64:
            return "float64";
        default:
            return "unknown";
    }
}

std::string to_string(Dtype::Code dtype) {
    return std::string(to_cstring(dtype));
}

}  // namespace p10
#include "dtype.hpp"

namespace p10 {
const char *to_cstring(Dtype::Code dtype) {
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
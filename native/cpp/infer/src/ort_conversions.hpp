#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>
#include <ptensor/dtype.hpp>
#include <ptensor/p10_result.hpp>

namespace p10::infer {
inline P10Result<Dtype> ort_to_hp3d_dtype(ONNXTensorElementDataType ort_dtype) {
    switch (ort_dtype) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return Ok<Dtype>(Dtype::Float32);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return Ok<Dtype>(Dtype::Uint8);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return Ok<Dtype>(Dtype::Int64);
        default:
            return Err(P10Error::InvalidArgument << "Unsupported ONNX data type");
    }
}

inline ONNXTensorElementDataType dtype_to_ort(Dtype dtype) {
    switch (dtype) {
        case Dtype::Float32:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        case Dtype::Uint8:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        case Dtype::Int64:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        default:
            (P10Error::InvalidArgument << "Unsupported HP3D data type").expect("Error");
    }

    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;  // Unreachable
}
}  // namespace p10::infer

#include "stack.hpp"

#include <ptensor/tensor.hpp>

namespace p10::op {
p10::P10Error stack(std::span<const p10::Tensor> inputs, int64_t axis, p10::Tensor& output) {
    if (inputs.size() == 0) {
        return P10Error::InvalidArgument << "No input tensors to stack";
    }

    if (inputs[0].device() != Device::Cpu) {
        return P10Error::InvalidArgument << "Stack operation only supports CPU tensors";
    }

    const auto& first_shape = inputs[0].shape();
    const auto first_dtype = inputs[0].dtype();

    const size_t dims = first_shape.dims();

    if (axis < 0) {
        axis += static_cast<int64_t>(dims) + 1;
    }

    if (axis < 0 || axis > static_cast<int64_t>(dims)) {
        return P10Error::InvalidArgument << "Axis out of bounds for stacking";
    }

    // Validate that all input tensors have the same shape
    for (size_t i = 1; i < inputs.size(); ++i) {
        if (inputs[i].shape() != first_shape) {
            return P10Error::InvalidArgument << "All input tensors must have the same shape";
        }
        if (inputs[i].dtype() != first_dtype) {
            return P10Error::InvalidArgument << "All input tensors must have the same dtype";
        }
        if (inputs[i].device() != Device::Cpu) {
            return P10Error::InvalidArgument << "Stack operation only supports CPU tensors";
        }
    }

    // Compute output shape
    auto shape_res = Shape::zeros(dims + 1);
    if (shape_res.is_error()) {
        return shape_res.error();
    }
    Shape out_shape = shape_res.unwrap();
    auto out_shape_s = out_shape.as_span();
    auto first_shape_s = first_shape.as_span();
    for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
        out_shape_s[i] = first_shape_s[i];
    }
    out_shape_s[axis] = static_cast<int64_t>(inputs.size());
    for (size_t i = axis + 1; i < dims + 1; ++i) {
        out_shape_s[i] = first_shape_s[i - 1];
    }

    // Create output tensor
    auto opts = TensorOptions().dtype(first_dtype).device(inputs[0].device());
    output = Tensor::empty(out_shape, opts).unwrap();

    // Perform stacking
    size_t inner_size = 1;
    for (size_t i = axis + 1; i < dims + 1; ++i) {
        inner_size *= out_shape_s[i];
    }
    size_t outer_size = 1;
    for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
        outer_size *= out_shape_s[i];
    }

    const size_t element_size = first_dtype.size_bytes();
    const size_t copy_size = inner_size * element_size;

    for (size_t outer = 0; outer < outer_size; ++outer) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            size_t out_offset = (outer * inputs.size() + i) * inner_size * element_size;
            size_t in_offset = outer * inner_size * element_size;
            std::memcpy(
                output.as_bytes().data() + out_offset,
                inputs[i].as_bytes().data() + in_offset,
                copy_size
            );
        }
    }
    return P10Error::Ok;
}
}  // namespace p10::op

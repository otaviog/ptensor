#pragma once

#include "ptensor/p10_error.hpp"

namespace p10 {
class Tensor;
}

namespace p10::op {
/// Resize a tensor using nearest-neighbor sampling.
///
/// Both input and output tensors use layout [C x H x W] (channels-first).
/// The output tensor is allocated internally; any existing content is overwritten.
/// Supports all dtypes. On x86/x86-64 with AVX2, contiguous uint8 tensors use
/// an optimized SIMD path.
///
/// # Arguments
///
/// * `input`: Source tensor with shape [C x H x W].
/// * `output`: Destination tensor; created with shape [C x new_height x new_width]
///   and the same dtype as `input`.
/// * `new_width`: Target width in pixels.
/// * `new_height`: Target height in pixels.
///
/// # Returns
///
/// * `P10Error::Ok` on success, or an error if the input shape is invalid.
P10Error resize(const Tensor& input, Tensor& output, size_t new_width, size_t new_height);
}  // namespace p10::op

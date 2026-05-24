#pragma once

#include <cstddef>
#include <utility>

namespace p10 {
class Tensor;
}

namespace p10::op {

/// Returns the arithmetic mean of `tensor` cast to double.
/// Returns NaN for an empty tensor.
double mean(const Tensor& tensor);

/// Returns `{value, flat_index}` of the smallest element in `tensor`.
/// Returns `{NaN, 0}` for an empty tensor.
std::pair<double, size_t> min(const Tensor& tensor);

/// Returns `{value, flat_index}` of the largest element in `tensor`.
/// Returns `{NaN, 0}` for an empty tensor.
std::pair<double, size_t> max(const Tensor& tensor);

}  // namespace p10::op

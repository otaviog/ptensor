#pragma once

#include <cstdint>
#include <numeric>
#include <span>

#include "detail/tensor_extents.hpp"
#include "p10_result.hpp"

namespace p10 {
class Shape;

/// A class representing the strides of a tensor.
class Stride: public detail::TensorExtents {
  public:
    static Stride from_contiguous_shape(const Shape& shape);

  // Constructor is private to enforce creation via factory methods,
  // ensuring correct stride initialization and invariants.
  private:
    Stride(size_t dims) : detail::TensorExtents {dims} {}

    using TensorExtents::TensorExtents;
};

inline P10Result<Stride> make_stride(const std::initializer_list<int64_t>& shape) {
    return detail::make_extent<Stride>(shape);
}

inline P10Result<Stride> make_stride(std::span<const int64_t> shape) {
    return detail::make_extent<Stride>(shape);
}

/// Converts a stride to a string.
inline std::string to_string(const Stride& stride) {
    return detail::to_string(stride);
}
}  // namespace p10

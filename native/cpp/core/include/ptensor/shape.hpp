#pragma once

#include <cstdint>
#include <numeric>
#include <span>

#include "detail/tensor_extents.hpp"
#include "ptensor_result.hpp"

namespace p10 {
class Shape: public detail::TensorExtents {
  public:
    int64_t count() const {
        const auto shape = as_span();
        return std::accumulate(shape.begin(), shape.end(), int64_t {1}, std::multiplies<int64_t>());
    }

    using TensorExtents::TensorExtents;
    friend class Stride;
};

inline PtensorResult<Shape> make_shape(const std::initializer_list<int64_t>& shape) {
    return detail::make_extent<Shape>(shape);
}

inline PtensorResult<Shape> make_shape(std::span<const int64_t> shape) {
    return detail::make_extent<Shape>(shape);
}

inline Shape make_shape(int64_t s0) {
    return make_shape({s0}).unwrap();
}

inline Shape make_shape(int64_t s0, int64_t s1) {
    return make_shape({s0, s1}).unwrap();
}

inline Shape make_shape(int64_t s0, int64_t s1, int64_t s2) {
    return make_shape({s0, s1, s2}).unwrap();
}

inline Shape make_shape(int64_t s0, int64_t s1, int64_t s2, int64_t s3) {
    return make_shape({s0, s1, s2, s3}).unwrap();
}

/// Converts a shape to a string.
inline std::string to_string(const Shape& shape) {
    return detail::to_string(shape);
}

/// A class representing the strides of a tensor.
class Stride: public detail::TensorExtents {
  public:
    Stride(const Shape& shape) : detail::TensorExtents {shape.dims()} {
        extent_[dims_ - 1] = 1;
        for (int i = int(dims_) - 2; i >= 0; i--) {
            extent_[i] = shape.extent_[i + 1] * extent_[i + 1];
        }
    }

  private:
    using TensorExtents::TensorExtents;
};

inline PtensorResult<Stride> make_stride(const std::initializer_list<int64_t>& shape) {
    return detail::make_extent<Stride>(shape);
}

inline PtensorResult<Stride> make_stride(std::span<const int64_t> shape) {
    return detail::make_extent<Stride>(shape);
}

/// Converts a stride to a string.
inline std::string to_string(const Stride& stride) {
    return detail::to_string(stride);
}
}  // namespace p10

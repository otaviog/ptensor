#pragma once

#include <cstdint>
#include <span>

#include "detail/tensor_extents.hpp"
#include "p10_result.hpp"

namespace p10 {
class Shape;

/// A class representing the strides of a tensor.
class Stride: public detail::TensorExtents {
  public:
    static Stride from_contiguous_shape(const Shape& shape);

    static P10Result<Stride> zeros(size_t dims) {
        if (dims < P10_MAX_SHAPE) {
            return Ok(Stride(dims));
        } else {
            return Err(
                P10Error::InvalidArgument,
                "Number of dimensions exceeds maximum supported shape"
            );
        }
    }

  private:
    explicit Stride(size_t dims) : TensorExtents(dims) {}

    /// Checks if the shape is empty, that is, it has no dimensions.
    bool empty() const {
        return dims_ == 0;
    }

  private:
    using TensorExtents::TensorExtents;
};

inline P10Result<Stride> make_stride(const std::initializer_list<int64_t>& shape) {
    return detail::make_extent<Stride>(shape);
}

inline P10Result<Stride> make_stride(std::span<const int64_t> shape) {
    return detail::make_extent<Stride>(shape);
}

inline Stride make_stride(int64_t s0) {
    return make_stride({s0}).unwrap();
}

inline Stride make_stride(int64_t s0, int64_t s1) {
    return make_stride({s0, s1}).unwrap();
}

inline Stride make_stride(int64_t s0, int64_t s1, int64_t s2) {
    return make_stride({s0, s1, s2}).unwrap();
}

inline Stride make_stride(int64_t s0, int64_t s1, int64_t s2, int64_t s3) {
    return make_stride({s0, s1, s2, s3}).unwrap();
}

/// Converts a stride to a string.
inline std::string to_string(const Stride& stride) {
    return detail::to_string(stride);
}
}  // namespace p10

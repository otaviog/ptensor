#pragma once

#include <cstdint>
#include <numeric>
#include <span>

#include "detail/tensor_extents.hpp"
#include "p10_result.hpp"
#include "ptensor/config.h"

namespace p10 {
class Shape: public detail::TensorExtents {
  public:
    static p10::P10Result<Shape> zeros(size_t dims) {
        if (dims < P10_MAX_SHAPE) {
            return Ok(Shape(dims));
        } else {
            return Err(
                P10Error::InvalidArgument,
                "Number of dimensions exceeds maximum supported shape"
            );
        }
    }

    int64_t count() const {
        if (dims_ == 0) {
            return 0;
        }
        const auto shape = as_span();
        return std::accumulate(shape.begin(), shape.end(), int64_t {1}, std::multiplies<int64_t>());
    }

    /// Checks if the shape is empty, that is, it has no dimensions.
    bool empty() const {
        return count() == 0;
    }

    Shape subshape(size_t start_dim, size_t end_dim = SIZE_MAX) const {
        return Shape(detail::TensorExtents::subextents(start_dim, end_dim));
    }

    P10Result<Shape> permute(const std::span<const size_t>& perm) const {
        return detail::TensorExtents::permute_extents(perm).map([](auto&& extents) {
            return Shape(std::forward<decltype(extents)>(extents));
        });
    }

    P10Result<Shape> transpose() const {
        if (dims_ != 2) {
            return Err(
                P10Error::InvalidArgument,
                "Transpose is only supported for 2D shapes"
            );
        }
        return Ok(Shape({extent_[1], extent_[0]}));
    }

    using TensorExtents::TensorExtents;
    friend class Stride;
};

inline P10Result<Shape> make_shape(const std::initializer_list<int64_t>& shape) {
    return detail::make_extent<Shape>(shape);
}

inline P10Result<Shape> make_shape(std::span<const int64_t> shape) {
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

inline Shape make_shape(int64_t s0, int64_t s1, int64_t s2, int64_t s3, int64_t s4) {
    return make_shape({s0, s1, s2, s3, s4}).unwrap();
}

/// Converts a shape to a string.
inline std::string to_string(const Shape& shape) {
    return detail::to_string(shape);
}

}  // namespace p10

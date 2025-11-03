#pragma once

#include <cstdint>
#include <numeric>
#include <span>

#include "detail/tensor_extents.hpp"
#include "p10_result.hpp"

namespace p10 {
class Shape: public detail::TensorExtents {
  public:
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

/// Converts a shape to a string.
inline std::string to_string(const Shape& shape) {
    return detail::to_string(shape);
}

}  // namespace p10

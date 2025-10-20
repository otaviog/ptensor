#pragma once

#include <array>
#include <span>
#include <sstream>

#include <ptensor/config.h>

#include "../p10_result.hpp"

namespace p10 {
namespace detail {
    class TensorExtents {
      public:
        using iterator = std::array<int64_t, P10_MAX_SHAPE>::iterator;
        using const_iterator = std::array<int64_t, P10_MAX_SHAPE>::const_iterator;

        TensorExtents() = default;

        /// The number of dimensions in the tensor.
        size_t dims() const {
            return dims_;
        }

        /// Checks if the shape is empty, that is, it has no dimensions.
        bool empty() const {
            return dims_ == 0;
        }

        std::span<int64_t> as_span() {
            return std::span<int64_t>(extent_.data(), dims_);
        }

        std::span<const int64_t> as_span() const {
            return std::span<const int64_t>(extent_.data(), dims_);
        }

        /// The size of the tensor along the given axis.
        ///
        /// # Arguments
        /// * `axis` - The axis index.
        ///
        /// # Returns
        /// * The size of the tensor along the given axis or an error if the axis is out of range.
        P10Result<int64_t> operator[](size_t axis) const {
            if (axis >= dims_) {
                return Err(P10Error::OutOfRange);
            }
            return Ok(int64_t {extent_[axis]});
        }

        bool operator==(const TensorExtents& other) const {
            return dims_ == other.dims_ && extent_ == other.extent_;
        }

        bool operator!=(const TensorExtents& other) const {
            return !(*this == other);
        }

        const int64_t* begin() const {
            return extent_.data();
        }

        const int64_t* end() const {
            return extent_.data() + dims_;
        }

      protected:
        TensorExtents(std::span<const int64_t> shape) {
            std::copy(shape.begin(), shape.end(), extent_.begin());
            dims_ = shape.size();
        }

        TensorExtents(size_t dims) : dims_(dims) {}

        template<typename iterator_t>
        TensorExtents(iterator_t begin_it, iterator_t end_it) :
            dims_ {size_t(std::distance(begin_it, end_it))} {
            std::copy(begin_it, end_it, extent_.begin());
        }

        std::array<int64_t, P10_MAX_SHAPE> extent_ {};
        size_t dims_ = 0;

        template<typename extent_t>
        friend P10Result<extent_t> make_extent(const std::initializer_list<int64_t>&);

        template<typename extent_t>
        friend P10Result<extent_t> make_extent(std::span<const int64_t> shape);
    };

    template<typename extent_t>
    P10Result<extent_t> make_extent(const std::initializer_list<int64_t>& shape) {
        if (shape.size() >= P10_MAX_SHAPE) {
            return Err(P10Error::OutOfRange);
        }
        return Ok(extent_t(shape.begin(), shape.end()));
    }

    template<typename extent_t>
    P10Result<extent_t> make_extent(std::span<const int64_t> shape) {
        if (shape.size() >= P10_MAX_SHAPE) {
            return Err(P10Error::OutOfRange);
        }
        return Ok(extent_t(shape));
    }

    inline std::string to_string(const TensorExtents& extents) {
        std::stringstream result;
        result << "[";
        for (size_t i = 0; i < extents.dims(); i++) {
            result << std::to_string(extents[i].unwrap());
            if (i < extents.dims() - 1) {
                result << ", ";
            }
        }
        result << "]";
        return result.str();
    }
}  // namespace detail
}  // namespace p10
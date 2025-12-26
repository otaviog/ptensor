#pragma once

#include <array>
#include <cstdint>
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

        TensorExtents(size_t dims) : dims_(dims) {}

        TensorExtents(TensorExtents&& other) noexcept {
            dims_ = other.dims_;
            extent_ = std::move(other.extent_);
            other.dims_ = 0;
            other.extent_.fill(0);
        }

        TensorExtents& operator=(TensorExtents&& other) noexcept {
            dims_ = other.dims_;
            extent_ = std::move(other.extent_);
            other.dims_ = 0;
            other.extent_.fill(0);
            return *this;
        }

        TensorExtents& operator=(const TensorExtents& other) = default;
        TensorExtents(const TensorExtents& other) = default;

        /// The number of dimensions in the tensor.
        size_t dims() const {
            return dims_;
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

        int64_t* begin() {
            return extent_.data();
        }

        int64_t* end() {
            return extent_.data() + dims_;
        }

        int64_t front() const {
            if (dims_ == 0) {
                return 0;
            }
            return extent_[0];
        }

        int64_t back() const {
            if (dims_ == 0) {
                return 0;
            }
            return extent_[dims_ - 1];
        }

        TensorExtents subextents(size_t start_dim, size_t end_dim) const {
            if (start_dim >= dims_ || start_dim >= end_dim) {
                return TensorExtents();
            }
            end_dim = end_dim > dims_ ? dims_ : end_dim;
            TensorExtents result;
            result.dims_ = end_dim - start_dim;
            for (size_t i = start_dim; i < end_dim; i++) {
                result.extent_[i - start_dim] = extent_[i];
            }
            return result;
        }

        P10Result<TensorExtents> permute_extents(const std::span<const size_t>& perm) const {
            if (perm.size() != dims_) {
                return Err(P10Error::OutOfRange << "Permutation size does not match number of dimensions");
            }

            TensorExtents result;
            result.dims_ = dims_;
            for (size_t i = 0; i < dims_; i++) {
                size_t perm_index = perm[i];
                result.extent_[i] = extent_[perm_index];
            }
            return Ok(std::move(result));
        }

      protected:
        TensorExtents(std::span<const int64_t> shape) {
            std::copy(shape.begin(), shape.end(), extent_.begin());
            dims_ = shape.size();
        }

        template<typename iterator_t>
        TensorExtents(iterator_t begin_it, iterator_t end_it) :
            dims_ {size_t(std::distance(begin_it, end_it))} {
            std::copy(begin_it, end_it, extent_.begin());
        }

        std::array<int64_t, P10_MAX_SHAPE> extent_ {0};
        size_t dims_ = 0;

        template<typename extent_t>
        friend P10Result<extent_t> make_extent(const std::initializer_list<int64_t>&);

        template<typename extent_t>
        friend P10Result<extent_t> make_extent(std::span<const int64_t> shape);
    };

    template<typename extent_t>
    P10Result<extent_t> make_extent(const std::initializer_list<int64_t>& shape) {
        if (shape.size() > P10_MAX_SHAPE) {
            return Err(P10Error::OutOfRange);
        }
        return Ok(extent_t(shape.begin(), shape.end()));
    }

    template<typename extent_t>
    P10Result<extent_t> make_extent(std::span<const int64_t> shape) {
        if (shape.size() > P10_MAX_SHAPE) {
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

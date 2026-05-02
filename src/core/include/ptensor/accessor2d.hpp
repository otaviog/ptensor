#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "accessor1d.hpp"

namespace p10 {
template<typename T>
class Accessor2D {
  public:
    Accessor2D(T* data, std::array<int64_t, 2> shape, std::array<int64_t, 2> strides) :
        data_(data),
        shape_(shape),
        strides_(strides) {}

    int64_t rows() const {
        return shape_[0];
    }

    int64_t cols() const {
        return shape_[1];
    }

    Accessor1D<T> operator[](size_t row) {
        assert(row < static_cast<size_t>(shape_[0]));
        return Accessor1D<T>(data_ + row * strides_[0], shape_[1], strides_[1]);
    }

    Accessor1D<const T> operator[](size_t row) const {
        assert(row < static_cast<size_t>(shape_[0]));
        return Accessor1D<const T>(data_ + row * strides_[0], shape_[1], strides_[1]);
    }

    Accessor2D<const T> as_const() const {
        return Accessor2D<const T>(data_, shape_, strides_);
    }

  private:
    T* data_;
    std::array<int64_t, 2> shape_;
    std::array<int64_t, 2> strides_;
};
}  // namespace p10
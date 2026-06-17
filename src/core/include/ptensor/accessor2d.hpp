#pragma once

#include <array>
#include <cassert>
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

    // A row is self-bounded to its own columns; halo comes from slicing a row of
    // a full-width plane (see Span2D), not from a region accessor.
    Accessor1D<T> operator[](int64_t row) {
        return Accessor1D<T>(data_ + row * strides_[0], shape_[1], strides_[1]);
    }

    Accessor1D<const T> operator[](int64_t row) const {
        return Accessor1D<const T>(data_ + row * strides_[0], shape_[1], strides_[1]);
    }

    Accessor2D<const T> as_const() const {
        return Accessor2D<const T>(data_, shape_, strides_);
    }

    Accessor2D transpose() const {
        return Accessor2D(data_, {shape_[1], shape_[0]}, {strides_[1], strides_[0]});
    }

  private:
    T* data_;
    std::array<int64_t, 2> shape_;
    std::array<int64_t, 2> strides_;
};
}  // namespace p10

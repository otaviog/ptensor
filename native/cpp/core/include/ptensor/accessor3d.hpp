#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>

#include "accessor2d.hpp"

namespace p10 {
template<typename T>
class Accessor3D {
  public:
    Accessor3D(T* data, std::array<int64_t, 3> shape, std::array<int64_t, 3> strides) :
        data_(data),
        shape_(shape),
        strides_(strides) {}

    int64_t channels() const {
        return shape_[0];
    }

    int64_t rows() const {
        return shape_[1];
    }

    int64_t cols() const {
        return shape_[2];
    }

    Accessor2D<T> operator[](size_t depth) {
        assert(depth < static_cast<size_t>(shape_[0]));
        return Accessor2D<T>(
            data_ + depth * strides_[0],
            {shape_[1], shape_[2]},
            {strides_[1], strides_[2]}
        );
    }

    Accessor2D<const T> operator[](size_t depth) const {
        assert(depth < static_cast<size_t>(shape_[0]));
        return Accessor2D<const T>(
            data_ + depth * strides_[0],
            {shape_[1], shape_[2]},
            {strides_[1], strides_[2]}
        );
    }

    Accessor3D<const T> as_const() const {
        return Accessor3D<const T>(data_, shape_, strides_);
    }

  private:
    T* data_;
    std::array<int64_t, 3> shape_;
    std::array<int64_t, 3> strides_;
};
}  // namespace p10
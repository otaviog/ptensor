#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>

namespace p10 {
template<typename T>
class Accessor1D {
  public:
    Accessor1D(T* data, int64_t size, int64_t stride) : data_(data), cols_(size), stride_(stride) {}

    T& operator[](size_t index) {
        assert(index < static_cast<size_t>(cols_));
        return *(data_ + index * stride_);
    }

    T operator[](size_t index) const {
        assert(index < static_cast<size_t>(cols_));
        return *(data_ + index * stride_);
    }

    int64_t cols() const {
        return cols_;
    }

    T* data() const {
        return data_;
    }

    // Contiguous view of the row. Only valid when the row is unit-stride, which
    // holds for rows of a contiguous buffer.
    std::span<T> as_span() const {
        assert(stride_ == 1);
        return std::span<T>(data_, static_cast<size_t>(cols_));
    }

    Accessor1D<const T> as_const() const {
        return Accessor1D<const T>(data_, cols_, stride_);
    }

  private:
    T* data_;
    int64_t cols_;
    int64_t stride_;
};
}  // namespace p10

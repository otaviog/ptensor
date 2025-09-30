#pragma once

#include <cassert>
#include <cstddef>

namespace p10 {
template<typename T>
class Span2D {
  public:
    Span2D() = default;

    Span2D(T* data, size_t height, size_t width) : data_(data), height_(height), width_(width) {}

    T* row(size_t row) {
        assert(row < height_);
        return data_ + row * width_;
    }

    const T* row(size_t row) const {
        assert(row < height_);
        return data_ + row * width_;
    }

    size_t height() const {
        return height_;
    }

    size_t width() const {
        return width_;
    }

    T& operator[](size_t index) {
        assert(index < height_ * width_);
        return data_[index];
    }

    const T& operator[](size_t index) const {
        assert(index < height_ * width_);
        return data_[index];
    }

  private:
    T* data_ = nullptr;
    size_t height_ = 0;
    size_t width_ = 0;
};
}  // namespace p10
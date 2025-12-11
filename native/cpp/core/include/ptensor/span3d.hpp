#pragma once

#include <cassert>
#include <cstddef>

#include "span2d.hpp"

namespace p10 {
template<typename T>
class Span3D {
  public:
    Span3D() = default;

    Span3D(T* data, size_t height, size_t width, size_t channels) :
        data_(data),
        height_(height),
        width_(width),
        channels_(channels) {}

    T* row(size_t row) {
        assert(row < height_);
        return data_ + row * width_ * channels_;
    }

    const T* row(size_t row) const {
        assert(row < height_);
        return data_ + row * width_ * channels_;
    }

    T* channel(size_t row, size_t col) {
        assert(row < height_);
        assert(col < width_);
        return data_ + row * width_ * channels_ + col * channels_;
    }

    const T* channel(size_t row, size_t col) const {
        assert(row < height_);
        assert(col < width_);
        return data_ + row * width_ * channels_ + col * channels_;
    }

    size_t height() const {
        return height_;
    }

    size_t width() const {
        return width_;
    }

    size_t channels() const {
        return channels_;
    }

    T* data() {
        return data_;
    }

    const T* data() const {
        return data_;
    }

  private:
    T* data_ = nullptr;
    size_t height_ = 0;
    size_t width_ = 0;
    size_t channels_ = 0;
};

template<typename T>
class PlanarSpan3D {
  public:
    PlanarSpan3D() = default;

    PlanarSpan3D(T* data, size_t channels, size_t height, size_t width) :
        data_(data),
        channels_(channels),
        height_(height),
        width_(width) {}

    Span2D<const T> operator[](size_t channel) const {
        assert(channel < channels_);
        return Span2D<const T>(data_ + width_ * height_ * channel, height_, width_);
    }

    Span2D<T> operator[](size_t channel) {
        assert(channel < channels_);
        return Span2D<T>(data_ + width_ * height_ * channel, height_, width_);
    }

    size_t channels() const {
        return channels_;
    }

    size_t height() const {
        return height_;
    }

    size_t width() const {
        return width_;
    }

  private:
    T* data_ = nullptr;
    size_t channels_ = 0;
    size_t height_ = 0;
    size_t width_ = 0;
};
}  // namespace p10
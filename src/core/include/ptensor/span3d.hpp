#pragma once

#include <cassert>
#include <cstdint>

#include "span2d.hpp"

namespace p10 {
template<typename T>
class Span3D {
  public:
    Span3D() = default;

    Span3D(T* data, int64_t height, int64_t width, int64_t channels) :
        data_(data),
        height_(height),
        width_(width),
        channels_(channels) {}

    T* row(int64_t row) {
        assert(row < height_);
        return data_ + row * width_ * channels_;
    }

    const T* row(int64_t row) const {
        assert(row < height_);
        return data_ + row * width_ * channels_;
    }

    T* channel(int64_t row, int64_t col) {
        assert(row < height_);
        assert(col < width_);
        return data_ + row * width_ * channels_ + col * channels_;
    }

    const T* channel(int64_t row, int64_t col) const {
        assert(row < height_);
        assert(col < width_);
        return data_ + row * width_ * channels_ + col * channels_;
    }

    int64_t height() const {
        return height_;
    }

    int64_t width() const {
        return width_;
    }

    int64_t channels() const {
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
    int64_t height_ = 0;
    int64_t width_ = 0;
    int64_t channels_ = 0;
};

template<typename T>
class PlanarSpan3D {
  public:
    PlanarSpan3D() = default;

    PlanarSpan3D(T* data, int64_t channels, int64_t height, int64_t width) :
        data_(data),
        channels_(channels),
        height_(height),
        width_(width) {}

    Span2D<const T> operator[](int64_t channel) const {
        assert(channel < channels_);
        return Span2D<const T>(data_ + width_ * height_ * channel, height_, width_);
    }

    Span2D<T> operator[](int64_t channel) {
        assert(channel < channels_);
        return Span2D<T>(data_ + width_ * height_ * channel, height_, width_);
    }

    int64_t channels() const {
        return channels_;
    }

    int64_t height() const {
        return height_;
    }

    int64_t width() const {
        return width_;
    }

  private:
    T* data_ = nullptr;
    int64_t channels_ = 0;
    int64_t height_ = 0;
    int64_t width_ = 0;
};
}  // namespace p10

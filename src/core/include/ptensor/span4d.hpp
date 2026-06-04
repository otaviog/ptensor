#pragma once

#include <cassert>
#include <cstddef>

#include "span3d.hpp"

namespace p10 {

template<typename T>
class Span4D {
  public:
    Span4D() = default;

    Span4D(T* data, size_t batch, size_t channels, size_t height, size_t width) :
        data_(data),
        batch_(batch),
        channels_(channels),
        height_(height),
        width_(width) {}

    PlanarSpan3D<T> operator[](size_t batch) {
        assert(batch < batch_);
        return PlanarSpan3D<T>(
            data_ + batch * channels_ * height_ * width_, channels_, height_, width_
        );
    }

    PlanarSpan3D<const T> operator[](size_t batch) const {
        assert(batch < batch_);
        return PlanarSpan3D<const T>(
            data_ + batch * channels_ * height_ * width_, channels_, height_, width_
        );
    }

    size_t batch() const {
        return batch_;
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
    size_t batch_ = 0;
    size_t channels_ = 0;
    size_t height_ = 0;
    size_t width_ = 0;
};

}  // namespace p10

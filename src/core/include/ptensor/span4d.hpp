#pragma once

#include <cassert>
#include <cstdint>

#include "span3d.hpp"

namespace p10 {

template<typename T>
class Span4D {
  public:
    Span4D() = default;

    Span4D(T* data, int64_t batch, int64_t channels, int64_t height, int64_t width) :
        data_(data),
        batch_(batch),
        channels_(channels),
        height_(height),
        width_(width) {}

    PlanarSpan3D<T> operator[](int64_t batch) {
        assert(batch < batch_);
        return PlanarSpan3D<T>(
            data_ + batch * channels_ * height_ * width_, channels_, height_, width_
        );
    }

    PlanarSpan3D<const T> operator[](int64_t batch) const {
        assert(batch < batch_);
        return PlanarSpan3D<const T>(
            data_ + batch * channels_ * height_ * width_, channels_, height_, width_
        );
    }

    int64_t batch() const {
        return batch_;
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
    int64_t batch_ = 0;
    int64_t channels_ = 0;
    int64_t height_ = 0;
    int64_t width_ = 0;
};

}  // namespace p10

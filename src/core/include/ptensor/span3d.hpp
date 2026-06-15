#pragma once

#include <cassert>
#include <cstdint>
#include "span2d.hpp"

namespace p10 {

template<typename T>
class Span3D {
  public:
    Span3D() = default;

    Span3D(T* data, int64_t channels, int64_t height, int64_t width) :
        data_(data),
        channels_(channels),
        rows_(height),
        cols_(width) {}

    Span2D<const T> operator[](int64_t channel) const {
        assert(channel < channels_);
        return Span2D<const T>(data_ + cols_ * rows_ * channel, rows_, cols_);
    }

    Span2D<T> operator[](int64_t channel) {
        assert(channel < channels_);
        return Span2D<T>(data_ + cols_ * rows_ * channel, rows_, cols_);
    }

    int64_t channels() const {
        return channels_;
    }

    int64_t rows() const {
        return rows_;
    }

    int64_t cols() const {
        return cols_;
    }
    
  private:
    T* data_ = nullptr;
    int64_t channels_ = 0;
    int64_t rows_ = 0;
    int64_t cols_ = 0;
};
}  // namespace p10

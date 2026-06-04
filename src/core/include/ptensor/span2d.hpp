#pragma once

#include <cassert>
#include <cstddef>
#include <span>

#include "accessor2d.hpp"

namespace p10 {
struct TileRegion2D {
    ptrdiff_t offset;
    size_t width;
    size_t height;
    size_t stride;
};

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

    std::span<const T> operator[](size_t row) const {
        assert(row < height_);
        return std::span<const T>(data_ + row * width_, width_);
    }

    std::span<T> operator[](size_t row) {
        assert(row < height_);
        return std::span<T>(data_ + row * width_, width_);
    }

    size_t height() const {
        return height_;
    }

    size_t width() const {
        return width_;
    }

    Accessor2D<const T> accessor(const TileRegion2D& region) const {
        return Accessor2D<T>(
            data_ + region.offset,
            {region.height, region.width},
            {region.stride, 1}
        );
    }

    Accessor2D<T> accessor(const TileRegion2D& region) {
        return Accessor2D<T>(
            data_ + region.offset,
            {region.height, region.width},
            {region.stride, 1}
        );
    }

  private:
    T* data_ = nullptr;
    size_t height_ = 0;
    size_t width_ = 0;
};
}  // namespace p10

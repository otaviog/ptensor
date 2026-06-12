#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>

#include "accessor2d.hpp"
#include "tile_region2d.hpp"

namespace p10 {

template<typename T>
class Span2D {
  public:
    Span2D() = default;

    Span2D(T* data, int64_t height, int64_t width) : data_(data), height_(height), width_(width) {}

    T* row(int64_t row) const {
        assert(row < height_);
        return data_ + row * width_;
    }

    std::span<T> operator[](int64_t row) const {
        assert(row < height_);
        return std::span<T>(data_ + row * width_, static_cast<size_t>(width_));
    }

    int64_t height() const {
        return height_;
    }

    int64_t width() const {
        return width_;
    }

    Accessor2D<T> operator()(const TileRegion2D& region) const {
        return Accessor2D<T>(
            data_ + region.row * width_ + region.col,
            {region.height, region.width},
            {width_, 1}
        );
    }

  private:
    T* data_ = nullptr;
    int64_t height_ = 0;
    int64_t width_ = 0;
};
}  // namespace p10

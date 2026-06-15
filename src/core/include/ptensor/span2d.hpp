#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>

#include "accessor2d.hpp"
#include "region2d.hpp"

namespace p10 {

template<typename T>
class Span2D {
  public:
    Span2D() = default;

    Span2D(T* data, int64_t rows, int64_t cols) : data_(data), rows_(rows), cols_(cols) {}

    int64_t rows() const {
        return rows_;
    }

    int64_t cols() const {
        return cols_;
    }
    
    std::span<const T> operator[](int64_t row) const {
        assert(row < rows_);
        return std::span<const T>(data_ + row * cols_, static_cast<size_t>(cols_));
    }

    std::span<T> operator[](int64_t row)  {
        assert(row < rows_);
        return std::span<T>(data_ + row * cols_, static_cast<size_t>(cols_));
    }
    
    Accessor2D<T> operator()(const Region2D& region) {
        return Accessor2D<T>(
            data_ + region.row * cols_ + region.col,
            {region.height, region.width},
            {cols_, 1}
        );
    }

    Accessor2D<const T> operator()(const Region2D& region) const {
        return Accessor2D<const T>(
            data_ + region.row * cols_ + region.col,
            {region.height, region.width},
            {cols_, 1}
        );
    }

  private:
    T* data_ = nullptr;
    int64_t rows_ = 0;
    int64_t cols_ = 0;
};
}  // namespace p10

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>

namespace p10 {
template<typename T>
class Accessor1D {
  public:
    // Bounds-carrying view: [min_index, max_index] is the inclusive index range
    // this view may address, in its own local coordinates (index 0 is data_).
    // It may reach outside [0, size) as long as it stays in range -- that is a
    // stencil halo, e.g. `row[-1]` when min_index < 0. left()/right() report how
    // far, so a border kernel can clamp a tap into [-left(), right()].
    Accessor1D(T* data, int64_t size, int64_t stride, int64_t min_index, int64_t max_index) :
        data_(data),
        cols_(size),
        stride_(stride),
        min_index_(min_index),
        max_index_(max_index) {}

    // Self-bounded: the view may only address its own elements (no halo). Used by
    // callers that slice a standalone buffer rather than a sub-region.
    Accessor1D(T* data, int64_t size, int64_t stride) :
        Accessor1D(data, size, stride, 0, size - 1) {}

    T& operator[](int64_t index) {
        assert(index >= min_index_ && index <= max_index_);
        return *(data_ + index * stride_);
    }

    T operator[](int64_t index) const {
        assert(index >= min_index_ && index <= max_index_);
        return *(data_ + index * stride_);
    }

    int64_t cols() const {
        return cols_;
    }

    T* data() const {
        return data_;
    }

    // Elements addressable below / above index 0; the valid index range is
    // [-left(), right()]. For a full-width view both are the view's own edges;
    // for a slice() of a row they reach to the row ends.
    int64_t left() const {
        return -min_index_;
    }

    int64_t right() const {
        return max_index_;
    }

    // Local-coordinate sub-view: index 0 of the result is index `offset` of this
    // view, `length` elements long, keeping the same reach (the index range just
    // shifts by `offset`). Used to hand a tile its own columns while letting it
    // still read the apron and clamp to the row edges.
    Accessor1D slice(int64_t offset, int64_t length) const {
        return Accessor1D(
            data_ + offset * stride_, length, stride_, min_index_ - offset, max_index_ - offset
        );
    }

    // Contiguous view of the row. Only valid when the row is unit-stride, which
    // holds for rows of a contiguous buffer.
    std::span<T> as_span() const {
        assert(stride_ == 1);
        return std::span<T>(data_, static_cast<size_t>(cols_));
    }

    Accessor1D<const T> as_const() const {
        return Accessor1D<const T>(data_, cols_, stride_, min_index_, max_index_);
    }

  private:
    T* data_;
    int64_t cols_;
    int64_t stride_;
    int64_t min_index_;
    int64_t max_index_;
};
}  // namespace p10

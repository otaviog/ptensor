#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace p10 {
template<typename T>
class Accessor1D {
  public:
    Accessor1D(T* data, int64_t size, int64_t stride) : data_(data), size_(size), stride_(stride) {}

    T& operator[](size_t index) {
        assert(index < static_cast<size_t>(size_));
        return *(data_ + index * stride_);
    }

    T operator[](size_t index) const {
        assert(index < static_cast<size_t>(size_));
        return *(data_ + index * stride_);
    }

    int64_t size() const {
        return size_;
    }

    T* data() const {
        return data_;
    }

  private:
    T* data_;
    int64_t size_;
    int64_t stride_;
};
}  // namespace p10
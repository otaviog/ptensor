#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <utility>

#include <ptensor/config.h>

namespace p10 {
template<typename scalar_t>
class Iterator {
  public:
    Iterator(scalar_t* data, std::span<const int64_t> shape, std::span<const int64_t> stride) :
        data_(data),
        shape_(shape),
        stride_(stride),
        total_elements_(
            std::accumulate(shape.begin(), shape.end(), int64_t {1}, std::multiplies<>())
        ) {}

    bool has_next() const {
        return curr_element_ < total_elements_;
    }

    /// Advances the iterator, returning a reference to the current element and
    /// its linear offset (in elements) into the underlying buffer.
    std::pair<scalar_t&, ptrdiff_t> next() {
        assert(has_next());
        curr_element_++;

        ptrdiff_t linear_index = 0;
        for (size_t i = 0; i < shape_.size(); i++) {
            linear_index += coords_[i] * stride_[i];
        }

        increase_coords();
        return {data_[linear_index], linear_index};
    }

    std::span<const int64_t> coords() const {
        return coords_;
    }

  private:
    void increase_coords() {
        for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
            if (coords_[i] + 1 < shape_[i]) {
                coords_[i]++;
                break;
            }
            coords_[i] = 0;
        }
    }

    scalar_t* data_;
    std::span<const int64_t> shape_;
    std::span<const int64_t> stride_;
    std::array<int64_t, P10_MAX_SHAPE> coords_ = {0};
    size_t curr_element_ = 0;
    size_t total_elements_ = 0;
};
}  // namespace p10

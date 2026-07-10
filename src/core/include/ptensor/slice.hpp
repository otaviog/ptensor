#pragma once

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <limits>

#include "p10_result.hpp"

namespace p10 {
/// Half-open range `[start, end)` over one tensor dimension, taken every
/// `step` elements. Negative `start`/`end` count from the end of the
/// dimension (Python style). `step` must be positive: negative strides are
/// not supported.
struct Slice {
  public:
    /// Sentinel for "until the end of the dimension".
    static constexpr int64_t END = std::numeric_limits<int64_t>::max();

    explicit Slice(int64_t start = 0, int64_t end = END, int64_t step = 1) :
        start(start),
        end(end),
        step(step) {}

    inline bool is_all() const;

    /// Resolves the slice against a dimension of `size`: negative indices are
    /// converted to absolute ones and the range is clamped to `[0, size]`.
    P10Result<Slice> resolve(int64_t size) const {
        if (step <= 0) {
            return Err(
                P10Error::InvalidArgument
                << "Slice step must be positive (negative strides are not supported)"
            );
        }
        int64_t abs_end = end;
        if (abs_end == END) {
            abs_end = size;
        } else if (abs_end < 0) {
            abs_end += size;
        }
        auto s = Slice {start < 0 ? size + start : start, abs_end, step};
        s.start = std::clamp(s.start, int64_t {0}, size);
        s.end = std::clamp(s.end, int64_t {0}, size);
        if (s.end <= s.start) {
            return Err(P10Error::InvalidArgument << "Slice range is empty");
        }
        return Ok(s);
    }

    bool operator==(const Slice&) const = default;

    int64_t start = 0;
    int64_t end = END;
    int64_t step = 1;
};

inline const Slice SLICE_ALL {0, Slice::END, 1};

inline bool Slice::is_all() const {
    return *this == SLICE_ALL;
}

template<typename T>
concept Sliceable = requires(T slice, int64_t size) {
    { slice.resolve(size) } -> std::same_as<P10Result<Slice>>;
};

}  // namespace p10

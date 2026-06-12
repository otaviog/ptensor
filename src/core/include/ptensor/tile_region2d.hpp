#pragma once

#include <cstdint>

namespace p10 {
// A rectangular sub-region of a 2D buffer, expressed in absolute (row, col)
// coordinates. The row stride is supplied by the buffer being indexed, so the
// same region can address buffers of different widths.
struct TileRegion2D {
    int64_t row;
    int64_t col;
    int64_t height;
    int64_t width;

    // Region with row/col and height/width swapped (the transposed location).
    TileRegion2D transposed() const {
        return {col, row, width, height};
    }
};
}  // namespace p10

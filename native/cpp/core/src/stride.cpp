
#include "stride.hpp"

#include "shape.hpp"

namespace p10 {

Stride Stride::from_contiguous_shape(const Shape& shape) {
    const size_t dims = shape.dims();
    Stride stride {dims};

    auto stride_span = stride.as_span();
    auto shape_span = shape.as_span();
    stride_span[dims - 1] = 1;
    for (int dim_index = int(dims) - 2; dim_index >= 0; dim_index--) {
        stride_span[dim_index] = shape_span[dim_index + 1] * stride_span[dim_index + 1];
    }
    return stride;
}
}  // namespace p10
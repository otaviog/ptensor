#include "tensor_scalar.hpp"

#include <ptensor/tensor.hpp>

namespace p10::op {
void multiply_scalar(Tensor& a, double scalar) {
    a.visit([=](auto span) {
        using T = typename decltype(span)::value_type;
        const T scalar_value = static_cast<T>(scalar);
        for (size_t i = 0; i < span.size(); ++i) {
            span[i] *= scalar_value;
        }
    });
}
}  // namespace p10::op
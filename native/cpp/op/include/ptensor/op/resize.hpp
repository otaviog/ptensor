
#include "p10_error.hpp"

namespace p10 {
class Tensor;
}

namespace p10::op {
P10Error resize(const Tensor& input, Tensor& output, size_t new_width, size_t new_height);
}
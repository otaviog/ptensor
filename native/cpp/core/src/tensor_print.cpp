#include "tensor_print.hpp"

#include <iomanip>

#include "tensor.hpp"

namespace p10 {
std::string to_string(const Tensor& tensor, const TensorStringOptions& options) {
    std::stringstream result;
    result << "Tensor(shape=" << to_string(tensor.shape())
           << ", dtype=" << to_string(tensor.dtype()) << ", values=[";

    size_t printed_elements = 0;
    size_t max_elements = options.max_elements().value_or(tensor.size());
    tensor.visit([&](auto span) {
        using SpanType = decltype(span)::value_type;
        for (size_t i = 0; i < tensor.size(); i++) {
            if (printed_elements >= max_elements) {
                result << "...";
                break;
            }
            if (i > 0) {
                result << ", ";
            }
            if constexpr (std::is_floating_point_v<SpanType>) {
                result << std::fixed << std::setprecision(options.float_precision()) << span[i];
            } else {
                result << span[i];
            }
            printed_elements++;
        }
    });

    result << "])";
    return result.str();
}
}  // namespace p10
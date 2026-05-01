#include "tensor_print.hpp"

#include <cstdint>
#include <iomanip>

#include "tensor.hpp"

namespace p10 {

template<typename T>
void to_string_1d_tensor(
    std::stringstream& ss,
    const T* data,
    Shape shape,
    Stride stride,
    const TensorStringOptions& options
) {
    ss << "[";

    const int64_t total_elements = shape[0].unwrap();
    const int64_t max_elements = options.max_elements().value_or(total_elements * 2);

    for (int64_t i = 0; i < total_elements; i++) {
        if (i > 0) {
            ss << ", ";
        }

        if (i >= max_elements / 2 && i < total_elements - max_elements / 2) {
            ss << "...";
            i = total_elements - max_elements / 2 - 1;
            continue;
        }

        const auto value = data[i * stride[0].unwrap()];

        if constexpr (std::is_floating_point_v<T>) {
            ss << std::fixed << std::setprecision(options.float_precision()) << value;
        } else {
            ss << value;
        }
    }
    ss << "]";
}

template<typename T>
void to_string_rec(
    std::stringstream& ss,
    const T* data,
    Shape shape,
    Stride stride,
    const TensorStringOptions& options
) {
    if (shape.dims() == 0) {
        ss << "[]";
        return;
    } else if (shape.dims() == 1) {
        return to_string_1d_tensor(ss, data, shape, stride, options);
    }
    ss << "[";
    const int64_t shape0 = shape[0].unwrap();
    const int64_t stride0 = stride[0].unwrap();
    for (int64_t i = 0; i < shape0; i++) {
        if (i > 0) {
            ss << ",\n";
        }
        to_string_rec(ss, data + i * stride0, shape.subshape(1), stride.substride(1), options);
    }
    ss << "]";
}

std::string to_string(const Tensor& tensor, const TensorStringOptions& options) {
    std::stringstream result;
    result << "Tensor(shape=" << to_string(tensor.shape())
           << ", dtype=" << to_string(tensor.dtype()) << ", values=";

    tensor.visit([&](auto span) {
        using SpanType = decltype(span)::value_type;

        to_string_rec<SpanType>(result, span.data(), tensor.shape(), tensor.stride(), options);
    });
    result << ')';
    return result.str();
}

}  // namespace p10
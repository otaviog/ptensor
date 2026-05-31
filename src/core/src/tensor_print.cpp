#include "tensor_print.hpp"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>

#include "base64.hpp"
#include "dtype.hpp"
#include "tensor.hpp"

namespace p10 {

namespace {

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
        }
        if (shape.dims() == 1) {
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

}  // namespace

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

std::string to_json(const Tensor& tensor) {
    return std::format(
        R"({{"dtype":"{}","shape":{},"stride":{},"blob":"{}"}})",
        to_string(tensor.dtype()),
        to_string(tensor.shape()),
        to_string(tensor.stride()),
        to_base64(tensor.as_bytes())
    );
}

namespace {
    thread_local std::string g_json_debug_buffer;
}  // namespace

const char* to_json_debug(const Tensor& tensor) {
    g_json_debug_buffer = to_json(tensor);
    return g_json_debug_buffer.c_str();
}

namespace {

    template<typename Reduce>
    double reduce_to_double(const Tensor& tensor, Reduce&& reduce) {
        if (tensor.empty() || tensor.size() == 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return tensor.visit([&](auto span) -> double {
            return std::forward<Reduce>(reduce)(span);
        });
    }

}  // namespace

double tensor_min_debug(const Tensor& tensor) {
    return reduce_to_double(tensor, [](auto span) {
        auto acc = static_cast<double>(span[0]);
        for (size_t i = 1; i < span.size(); ++i) {
            acc = std::min(acc, static_cast<double>(span[i]));
        }
        return acc;
    });
}

double tensor_max_debug(const Tensor& tensor) {
    return reduce_to_double(tensor, [](auto span) {
        auto acc = static_cast<double>(span[0]);
        for (size_t i = 1; i < span.size(); ++i) {
            acc = std::max(acc, static_cast<double>(span[i]));
        }
        return acc;
    });
}

double tensor_mean_debug(const Tensor& tensor) {
    return reduce_to_double(tensor, [](auto span) {
        double sum = 0.0;
        for (size_t i = 0; i < span.size(); ++i) {
            sum += static_cast<double>(span[i]);
        }
        return sum / static_cast<double>(span.size());
    });
}

}  // namespace p10

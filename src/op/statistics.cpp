#include "statistics.hpp"

#include <limits>
#include <vector>

#include <ptensor/tensor.hpp>

namespace p10::op {

namespace {
    constexpr double NAN_VALUE = std::numeric_limits<double>::quiet_NaN();

    bool is_empty(const Tensor& tensor) {
        return tensor.empty() || tensor.size() == 0;
    }
}  // namespace

double mean(const Tensor& tensor) {
    if (is_empty(tensor)) {
        return NAN_VALUE;
    }
    return tensor.visit([](auto span) -> double {
        double sum = 0.0;
        for (const auto value : span) {
            sum += static_cast<double>(value);
        }
        return sum / static_cast<double>(span.size());
    });
}

P10Error mean(const Tensor& tensor, int64_t axis, Tensor& mean) {
    if (tensor.device() != Device::Cpu) {
        return P10Error::NotImplemented << "Axis mean is only implemented for CPU tensors";
    }
    if (!tensor.is_contiguous()) {
        return P10Error::NotImplemented << "Axis mean is only implemented for contiguous tensors";
    }
    if (is_empty(tensor)) {
        return P10Error::InvalidArgument << "Cannot compute mean of an empty tensor";
    }

    const auto dims = static_cast<int64_t>(tensor.dims());
    if (axis < 0) {
        axis += dims;
    }
    if (axis < 0 || axis >= dims) {
        return P10Error::InvalidArgument << "Axis is out of range";
    }

    const auto shape = tensor.shape().as_span();
    size_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer *= static_cast<size_t>(shape[i]);
    }
    const auto axis_size = static_cast<size_t>(shape[axis]);
    size_t inner = 1;
    for (int64_t i = axis + 1; i < dims; ++i) {
        inner *= static_cast<size_t>(shape[i]);
    }

    // Output shape is the input shape with the reduced axis removed. A fully
    // reduced 1D tensor collapses to a single scalar element.
    std::vector<int64_t> out_dims;
    for (int64_t i = 0; i < dims; ++i) {
        if (i != axis) {
            out_dims.push_back(shape[i]);
        }
    }
    if (out_dims.empty()) {
        out_dims.push_back(1);
    }

    auto out_shape = make_shape(std::span<const int64_t>(out_dims));
    if (out_shape.is_error()) {
        return out_shape.error();
    }
    P10_RETURN_IF_ERROR(mean.create(out_shape.unwrap(), Dtype::Float64));
    auto out_span = mean.as_span1d<double>().unwrap();

    return tensor.visit([&](auto in_span) -> P10Error {
        for (size_t o = 0; o < outer; ++o) {
            for (size_t i = 0; i < inner; ++i) {
                double sum = 0.0;
                for (size_t k = 0; k < axis_size; ++k) {
                    sum += static_cast<double>(in_span[(o * axis_size + k) * inner + i]);
                }
                out_span[o * inner + i] = sum / static_cast<double>(axis_size);
            }
        }
        return P10Error::Ok;
    });
}

std::pair<double, size_t> min(const Tensor& tensor) {
    if (is_empty(tensor)) {
        return {NAN_VALUE, 0};
    }
    return tensor.visit([](auto span) -> std::pair<double, size_t> {
        auto best = static_cast<double>(span[0]);
        size_t best_idx = 0;
        for (size_t i = 1; i < span.size(); ++i) {
            const auto value = static_cast<double>(span[i]);
            if (value < best) {
                best = value;
                best_idx = i;
            }
        }
        return {best, best_idx};
    });
}

std::pair<double, size_t> max(const Tensor& tensor) {
    if (is_empty(tensor)) {
        return {NAN_VALUE, 0};
    }
    return tensor.visit([](auto span) -> std::pair<double, size_t> {
        auto best = static_cast<double>(span[0]);
        size_t best_idx = 0;
        for (size_t i = 1; i < span.size(); ++i) {
            const auto value = static_cast<double>(span[i]);
            if (value > best) {
                best = value;
                best_idx = i;
            }
        }
        return {best, best_idx};
    });
}

}  // namespace p10::op

#include "statistics.hpp"

#include <limits>

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

#pragma once

#include <optional>
#include <string>

namespace p10 {
class Tensor;

class TensorStringOptions {
  public:
    int float_precision() const {
        return float_precision_;
    }

    /// Sets the number of decimal places for floating point numbers.
    TensorStringOptions& float_precision(int precision) {
        float_precision_ = precision;
        return *this;
    }

    /// Gets the maximum number of elements to print.
    std::optional<size_t> max_elements() const {
        return max_elements_;
    }

    /// Sets the maximum number of elements to print.
    TensorStringOptions& max_elements(size_t max_elements) {
        max_elements_ = max_elements;
        return *this;
    }

  private:
    int float_precision_ = 4;
    std::optional<size_t> max_elements_;
};

std::string to_string(const Tensor& tensor, const TensorStringOptions& options = {});

std::string to_json(const Tensor& tensor);

/// Encodes `tensor` as JSON into a per-thread static buffer and returns a
/// pointer to its contents. The static buffer keeps the string alive past the
/// debugger's `evaluate` call so the result string can be read directly from
/// the response (no `readMemory` needed for typical sizes).
const char* to_json_debug(const Tensor& tensor);

/// Returns the minimum value of `tensor` cast to double. Empty tensors return
/// NaN. Designed as a hook for debugger visualizers (e.g. natvis intrinsics).
double tensor_min_debug(const Tensor& tensor);

/// Returns the maximum value of `tensor` cast to double. Empty tensors return
/// NaN. Designed as a hook for debugger visualizers.
double tensor_max_debug(const Tensor& tensor);

/// Returns the arithmetic mean of `tensor` cast to double. Empty tensors
/// return NaN. Designed as a hook for debugger visualizers.
double tensor_mean_debug(const Tensor& tensor);

}  // namespace p10

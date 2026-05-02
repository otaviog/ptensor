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
}  // namespace p10
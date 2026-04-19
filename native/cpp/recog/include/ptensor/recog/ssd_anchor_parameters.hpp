#pragma once

#include <span>
#include <initializer_list>
#include <vector>

namespace p10::recog {

class SsdAnchorParameters {
  public:
    SsdAnchorParameters(
        std::initializer_list<std::initializer_list<size_t>> min_sizes,
        std::initializer_list<size_t> steps,
        float center_variance,
        float size_variance
    ) : min_sizes_(min_sizes.begin(), min_sizes.end()),
        steps_(steps.begin(), steps.end()),
        center_variance_(center_variance),
        size_variance_(size_variance) {}

    std::span<const std::vector<size_t>> min_sizes() const { return min_sizes_; }
    std::span<const size_t> steps() const { return steps_; }
    float center_variance() const { return center_variance_; }
    float size_variance() const { return size_variance_; }

  private:
    std::vector<std::vector<size_t>> min_sizes_;
    std::vector<size_t> steps_;
    float center_variance_;
    float size_variance_;
};

}  // namespace p10::recog

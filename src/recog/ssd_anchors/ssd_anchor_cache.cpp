#include "ssd_anchor_cache.hpp"

#include <cassert>

namespace p10::recog {

std::span<const SsdAnchorRect>
SsdAnchorCache::get_anchors(size_t image_width, size_t image_height) {
    const auto incoming_size = std::make_pair(image_width, image_height);
    if (image_size_ != incoming_size) {
        image_size_ = incoming_size;
        generate(image_width, image_height);
    }
    return std::span(anchors_.data(), anchors_.size());
}

void SsdAnchorCache::generate(size_t image_width, size_t image_height) {
    anchors_.clear();
    const auto steps = params_.steps();
    const auto min_sizes = params_.min_sizes();
    for (size_t idx = 0; idx < steps.size(); ++idx) {
        const auto step = steps[idx];
        const size_t step_width = image_width / static_cast<int>(step);
        const size_t step_height = image_height / static_cast<int>(step);

        for (size_t row_idx = 0; row_idx < step_height; ++row_idx) {
            for (size_t col_idx = 0; col_idx < step_width; ++col_idx) {
                for (const auto min_size : min_sizes[idx]) {
                    anchors_.push_back({
                        .cx = (static_cast<float>(col_idx) + 0.5f) * static_cast<float>(step)
                            / static_cast<float>(image_width),
                        .cy = (static_cast<float>(row_idx) + 0.5f) * static_cast<float>(step)
                            / static_cast<float>(image_height),
                        .anchor_w = static_cast<float>(min_size) / static_cast<float>(image_width),
                        .anchor_h = static_cast<float>(min_size) / static_cast<float>(image_height),
                    });
                }
            }
        }
    }
}

}  // namespace p10::recog

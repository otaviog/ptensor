#pragma once

#include <span>
#include <utility>
#include <vector>

#include "ssd_anchor_decoder.hpp"
#include "ssd_anchor_parameters.hpp"

namespace p10::recog {

class SsdAnchorCache {
  public:
    explicit SsdAnchorCache(SsdAnchorParameters params) : params_(std::move(params)) {}

    std::span<const SsdAnchorRect> get_anchors(size_t image_width, size_t image_height);

    SsdAnchorDecoder decoder() const {
        return SsdAnchorDecoder(params_.center_variance(), params_.size_variance());
    }

  private:
    void generate(size_t image_width, size_t image_height);

    SsdAnchorParameters params_;
    std::pair<size_t, size_t> image_size_ {0, 0};
    std::vector<SsdAnchorRect> anchors_;
};

}  // namespace p10::recog

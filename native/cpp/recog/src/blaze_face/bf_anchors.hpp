#pragma once

#include <span>
#include <utility>
#include <vector>

#include "geom/rect2.hpp"

namespace p10::recog {

struct BfAnchorRect {
    float cx, cy;  // anchor center, normalized to [0, 1]
    float anchor_w, anchor_h;  // anchor dimensions, normalized to [0, 1]
};

class BfAnchorsParameters {
  public:
    std::vector<std::vector<size_t>> min_sizes;
    std::vector<size_t> steps;
    float center_variance;  // scales center offset (cx/cy decoding)
    float size_variance;  // scales log-space size delta (w/h decoding)

    void generate(size_t image_width, size_t image_height, std::vector<BfAnchorRect>& priors) const;
};

class BfAnchorDecoder {
  public:
    BfAnchorDecoder(float center_variance, float size_variance) :
        center_variance_(center_variance),
        size_variance_(size_variance) {}

    Rect2f decode_rect(std::span<const float> box, const BfAnchorRect& anchor) const;

    void decode_landmarks(
        std::span<const float> landmarks,
        const BfAnchorRect& anchor,
        std::span<Point2f> out_landmarks
    ) const;

  private:
    float center_variance_, size_variance_;
};

class BfAnchorCache {
  public:
    explicit BfAnchorCache(BfAnchorsParameters params) : params_(std::move(params)) {}

    std::span<const BfAnchorRect> get_anchors(size_t image_width, size_t image_height);

    BfAnchorDecoder decoder() const {
        return BfAnchorDecoder(params_.center_variance, params_.size_variance);
    }

  private:
    BfAnchorsParameters params_;
    std::pair<size_t, size_t> image_size_ {};
    std::vector<BfAnchorRect> anchors_;
};

}  // namespace p10::recog

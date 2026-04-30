#pragma once
#include <span>

#include "geom/rect2.hpp"
#include "ssd_anchor_rect.hpp"

namespace p10::recog {
class SsdAnchorDecoder {
  public:
    SsdAnchorDecoder(float center_variance, float size_variance) :
        center_variance_(center_variance),
        size_variance_(size_variance) {}

    Rect2f decode_rect(std::span<const float> box, const SsdAnchorRect& anchor) const;

    void decode_landmarks(
        std::span<const float> landmarks,
        const SsdAnchorRect& anchor,
        std::span<Point2f> out_landmarks
    ) const;

  private:
    float center_variance_, size_variance_;
};
}  // namespace p10::recog

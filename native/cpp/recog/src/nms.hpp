#pragma once

#include <span>
#include <vector>

#include "geom/rect2.hpp"

namespace p10::recog {
class Nms {
  public:
    Nms(float iou_threshold) : iou_threshold_(iou_threshold) {}
    
    void filter_nms(
        std::span<const Rect2f> rects,
        std::span<const float> scores,
        std::vector<size_t> &selected
    );
    
  private:
    float iou_threshold_;
};
}  // namespace p10::recog

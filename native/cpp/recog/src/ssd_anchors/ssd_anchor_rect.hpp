#pragma once

namespace p10::recog {

struct SsdAnchorRect {
    float cx, cy;  // anchor center, normalized to [0, 1]
    float anchor_w, anchor_h;  // anchor dimensions, normalized to [0, 1]
};

}  // namespace p10::recog

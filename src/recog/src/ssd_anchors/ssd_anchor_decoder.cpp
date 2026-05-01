#include "ssd_anchor_decoder.hpp"

#include <cassert>

namespace p10::recog {
Rect2f
SsdAnchorDecoder::decode_rect(std::span<const float> box, const SsdAnchorRect& anchor) const {
    const float cx_delta = box[0];
    const float cy_delta = box[1];
    const float w_delta = box[2];
    const float h_delta = box[3];

    const float width = anchor.anchor_w * std::exp(w_delta * size_variance_);
    const float height = anchor.anchor_h * std::exp(h_delta * size_variance_);

    const float x_start = anchor.cx + cx_delta * center_variance_ * anchor.anchor_w - width * 0.5f;
    const float y_start = anchor.cy + cy_delta * center_variance_ * anchor.anchor_h - height * 0.5f;

    return Rect2f({x_start, y_start}, {x_start + width, y_start + height});
}

void SsdAnchorDecoder::decode_landmarks(
    std::span<const float> landmarks,
    const SsdAnchorRect& anchor,
    std::span<Point2f> out_landmarks
) const {
    assert(landmarks.size() == out_landmarks.size() * 2);
    size_t out_lm_idx = 0;
    for (size_t point_idx = 0; point_idx < landmarks.size(); point_idx += 2, out_lm_idx++) {
        const float x = landmarks[point_idx];
        const float y = landmarks[point_idx + 1];
        out_landmarks[out_lm_idx] = Point2f {
            anchor.cx + x * center_variance_ * anchor.anchor_w,
            anchor.cy + y * center_variance_ * anchor.anchor_h
        };
    }
}
}  // namespace p10::recog

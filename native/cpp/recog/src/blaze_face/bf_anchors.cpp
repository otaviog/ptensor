#include "bf_anchors.hpp"

#include <cassert>
#include <cmath>

namespace p10::recog {

void BfAnchorsParameters::generate(
    size_t image_width,
    size_t image_height,
    std::vector<BfAnchorRect>& priors
) const {
    priors.clear();
    for (size_t idx = 0; idx < steps.size(); ++idx) {
        const auto step = steps[idx];
        const size_t step_width = image_width / int(step);
        const size_t step_height = image_height / int(step);

        for (size_t row_idx = 0; row_idx < step_height; ++row_idx) {
            for (size_t col_idx = 0; col_idx < step_width; ++col_idx) {
                for (const auto min_size : min_sizes[idx]) {
                    priors.push_back({
                        .cx = (float(col_idx) + 0.5f) * float(step) / float(image_width),
                        .cy = (float(row_idx) + 0.5f) * float(step) / float(image_height),
                        .anchor_w = float(min_size) / float(image_width),
                        .anchor_h = float(min_size) / float(image_height),
                    });
                }
            }
        }
    }
}

Rect2f BfAnchorDecoder::decode_rect(std::span<const float> box, const BfAnchorRect& anchor) const {
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

void BfAnchorDecoder::decode_landmarks(
    std::span<const float> landmarks,
    const BfAnchorRect& anchor,
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

std::span<const BfAnchorRect> BfAnchorCache::get_anchors(size_t image_width, size_t image_height) {
    const auto incoming_size = std::make_pair(image_width, image_height);
    if (image_size_ != incoming_size) {
        image_size_ = incoming_size;
        params_.generate(image_width, image_height, anchors_);
    }
    return std::span(anchors_.data(), anchors_.size());
}

}  // namespace p10::recog

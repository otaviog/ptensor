#pragma once

#include "../nms.hpp"
#include "bf_anchors.hpp"
#include "face_detection.hpp"

namespace p10::recog {

class BfPostprocess {
  public:
    BfPostprocess(const BfAnchorsParameters& anchor_params, Nms nms);

    void process(
        size_t input_width,
        size_t input_height,
        float preprocess_scale_ratio,
        std::span<const Tensor> model_outputs,
        std::span<FaceDetection> detections
    );

  private:
    std::vector<Rect2f> rect_buffer_;
    std::vector<float> conf_buffer_;
    std::vector<Point2f> landmark_buffer_;
    std::vector<size_t> selected_;
    std::vector<size_t> row_index_buffer_;

    float conf_threshold_ = 0.5f;
    Nms nms_;

    BfAnchorCache anchors_;
};

}  // namespace p10::recog

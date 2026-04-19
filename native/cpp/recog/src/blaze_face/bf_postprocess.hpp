#pragma once

#include "../nms.hpp"
#include "../ssd_anchors/ssd_anchor_cache.hpp"
#include "face_detection.hpp"
#include "ssd_anchor_parameters.hpp"

namespace p10::recog {

class BfPostprocess {
  public:
    BfPostprocess(
        const SsdAnchorParameters& anchor_params,
        float nms_iou_threshold,
        float conf_threshold
    ) :
        anchors_(anchor_params),
        nms_(nms_iou_threshold),
        conf_threshold_(conf_threshold) {}

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

    SsdAnchorCache anchors_;
    Nms nms_;
    float conf_threshold_ = 0.5f;
};

}  // namespace p10::recog

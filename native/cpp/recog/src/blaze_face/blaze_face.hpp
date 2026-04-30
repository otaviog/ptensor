#pragma once

#include <array>
#include <memory>

#include <ptensor/tensor.hpp>

#include "bf_postprocess.hpp"
#include "bf_preprocess.hpp"
#include "face_detection.hpp"

namespace p10::infer {
class IInfer;
}

namespace p10::recog {

class BlazeFace: public IFaceDetector {
  public:
    BlazeFace(
        infer::IInfer* infer,
        size_t target_size,
        const SsdAnchorParameters& anchor_params,
        float nms_iou_threshold,
        float threshold
    ) :
        infer_(infer),
        pre_process_(target_size),
        post_process_(anchor_params, nms_iou_threshold, threshold) {}

    ~BlazeFace() = default;

    P10Error detect(Tensor& images, std::span<FaceDetection> out_detections) override;

  private:
    std::unique_ptr<infer::IInfer> infer_;

    BfPreprocessing pre_process_;
    std::array<Tensor, 1> input_buffer_;

    BfPostprocess post_process_;
    std::array<Tensor, 3> outputs_;
};
}  // namespace p10::recog

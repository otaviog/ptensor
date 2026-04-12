#pragma once

#include <ptensor/tensor.hpp>
#include <array>
#include <memory>

#include "face_detection.hpp"

#include "bf_preprocess.hpp"
#include "bf_postprocess.hpp"

namespace p10::infer {
class IInfer;
}

namespace p10::recog {

class BlazeFace: public IFaceDetector {
  public:
    BlazeFace(infer::IInfer* infer);
    ~BlazeFace();

    P10Error detect(Tensor& images, std::span<FaceDetection> out_detections) override;

  private:
    std::unique_ptr<infer::IInfer> infer_;

    size_t target_size_ = 224;
    float threshold_ = 0.7;
    
    BfPreprocessing pre_process_;
    std::array<Tensor, 1> input_buffer_;

    BfPostprocess post_process_;
    std::array<Tensor, 3> outputs_;
};
}  // namespace p10::recog

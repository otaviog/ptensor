#pragma once

#include <vector>

#include <ptensor/p10_error.hpp>

#include "geom/point2.hpp"
#include "geom/rect2.hpp"
#include "ptensor/infer/infer.hpp"

namespace p10 {
class Tensor;
}

namespace p10::recog {
struct FaceDetection {
    std::vector<Rect2i> faces;
    std::vector<float> confidences;
    std::vector<std::vector<Point2i>> landmarks;

    void clear() {
        faces.clear();
        confidences.clear();
        landmarks.clear();
    }
};

class FaceDetectorConfig {
  public:
    enum Model { BlazeFace };

    FaceDetectorConfig& model(Model model) {
        model_ = model;
        return *this;
    }

    Model model() const {
        return model_;
    }

  private:
    Model model_;
};

class IFaceDetector {
  public:
    static P10Result<IFaceDetector*> create(FaceDetectorConfig config, infer::IInfer* infer_engine);

    virtual ~IFaceDetector() {}

    /// Detect faces
    ///
    /// # Arguments
    ///
    /// * images: images in the [N x C x H x W] format
    virtual P10Error detect(Tensor& images, std::span<FaceDetection> out_detections) = 0;

  private:
};
}  // namespace p10::recog

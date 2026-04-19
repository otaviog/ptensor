#pragma once

#include <span>
#include <variant>
#include <vector>

#include <ptensor/p10_result.hpp>

#include "geom/point2.hpp"
#include "geom/rect2.hpp"
#include "ssd_anchor_parameters.hpp"

namespace p10 {
class Tensor;
}

namespace p10::infer {
class IInfer;
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

class BlazeFaceModel {
  public:
    const SsdAnchorParameters& ssd_params() const {
        return ssd_params_;
    }

    size_t image_size() const {
        return image_size_;
    }

    BlazeFaceModel& image_size(size_t size) {
        image_size_ = size;
        return *this;
    }

    BlazeFaceModel& ssd_params(const SsdAnchorParameters& params) {
        ssd_params_ = params;
        return *this;
    }

    float nms_iou_threshold() const {
        return nms_iou_threshold_;
    }

    BlazeFaceModel& nms_iou_threshold(float threshold) {
        nms_iou_threshold_ = threshold;
        return *this;
    }

    float threshold() const {
        return score_threshold_;
    }

    BlazeFaceModel& threshold(float threshold) {
        score_threshold_ = threshold;
        return *this;
    }

  private:
    size_t image_size_ = 1280;
    SsdAnchorParameters ssd_params_ = {
        {{8, 11}, {14, 19, 26, 38, 64, 149}},  // min_sizes
        {8, 16},  // steps
        0.1,  // center_variance
        0.2  // size_variance
    };
    float nms_iou_threshold_ = 0.3f;
    float score_threshold_ = 0.95f;
};

using FaceDetectorModel = std::variant<BlazeFaceModel>;

/// Interface for face detection
class IFaceDetector {
  public:
    /// Create a face detector
    ///
    /// # Arguments
    ///
    /// * config: the face model and its configuration
    /// * infer_engine: inference engine to use for the face detector
    static P10Result<IFaceDetector*>
    create(const FaceDetectorModel& model, infer::IInfer* infer_engine);

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

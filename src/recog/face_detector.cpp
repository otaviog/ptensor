#include "face_detector.hpp"

#include <ptensor/infer/infer.hpp>

#include "blaze_face/blaze_face.hpp"

namespace p10::recog {

P10Result<std::unique_ptr<IFaceDetector>>
IFaceDetector::create(const FaceDetectorModel& model, std::unique_ptr<infer::IInfer> infer_engine) {
    if (std::holds_alternative<BlazeFaceModel>(model)) {
        const auto& blaze_face_config = std::get<BlazeFaceModel>(model);
        return Ok<std::unique_ptr<IFaceDetector>>(std::unique_ptr<IFaceDetector>(new BlazeFace(
            std::move(infer_engine),
            blaze_face_config.image_size(),
            blaze_face_config.ssd_params(),
            blaze_face_config.nms_iou_threshold(),
            blaze_face_config.threshold()
        )));
    }
    return Err(P10Error::InvalidArgument << "Unsupported face detection model");
}

P10Error IFaceDetector::verify_detect_arguments(Tensor& images, std::span<FaceDetection> out_detections) {
    const auto input_shape = images.shape().as_span();
    if (input_shape.size() != 4) {
        return P10Error::InvalidArgument << "Input tensor must have 4 dimensions [N x C x H x W]";
    }
    if (input_shape[1] != 3) {
        return P10Error::InvalidArgument << "Input tensor must have 3 channels (C=3)";
    }

    if (input_shape[0] != out_detections.size()) {
        return P10Error::InvalidArgument
            << "Output detections size must match the number of input images";
    }

}

}  // namespace p10::recog

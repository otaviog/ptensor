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

}  // namespace p10::recog

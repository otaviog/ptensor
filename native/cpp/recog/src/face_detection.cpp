#include "face_detection.hpp"

#include "blaze_face/blaze_face.hpp"
#include "ptensor/infer/infer.hpp"

namespace p10::recog {
P10Result<IFaceDetector*>
IFaceDetector::create(const FaceDetectorModel &model, infer::IInfer* infer_engine) {
    if (std::holds_alternative<BlazeFaceModel>(model)) {
        const auto& blaze_face_config = std::get<BlazeFaceModel>(model);
        return Ok<IFaceDetector*>(new BlazeFace(
            infer_engine,
            blaze_face_config.image_size(),
            blaze_face_config.ssd_params(),
            blaze_face_config.nms_iou_threshold(),
            blaze_face_config.threshold()
                                      ));
    }
    return Err(P10Error::InvalidArgument << "Unsupported face detection model");
}

}  // namespace p10::recog

#include "face_detection.hpp"

#include "blaze_face/blaze_face.hpp"
#include "ptensor/infer/infer.hpp"

namespace p10::recog {
P10Result<IFaceDetector*> IFaceDetector::create(FaceDetectorConfig config, infer::IInfer* infer_engine) {
    switch (config.model()) {
        case FaceDetectorConfig::BlazeFace:
            return Ok(new recog::BlazeFace(infer_engine));
            break;
    default:
        return Err(P10Error::InvalidArgument << "Unsupported face detection model");
    }
}

}  // namespace p10::recog

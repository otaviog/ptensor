#include <catch2/catch_test_macros.hpp>
#include <ptensor/infer/infer.hpp>
#include <ptensor/infer/infer_config.hpp>
#include <ptensor/io/image.hpp>

#include "face_detection.hpp"

namespace p10::recog {
TEST_CASE("recog::FaceDetection::blaze_face", "[recog][face][blaze_face]") {
    IFaceDetector* detector = IFaceDetector::create(
        FaceDetectorConfig::BlazeFace,
        infer::IInfer::from_onnx(
            "tests/data/face_detectors/blazefaces-320.onnx",
            infer::InferConfig()).expect("should load model")
        ).expect("Cant create detector");

    auto image = io::load_image("tests/data/face_detectors/faces.jpg").expect("should load image");

    std::array<FaceDetection, 1> detections;
    detector->detect(image, detections).expect("Face detected");
    REQUIRE(detections[0].faces.size() == 4);
}

}  // namespace p10::recog

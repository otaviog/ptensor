#include <catch2/catch_test_macros.hpp>
#include <ptensor/infer/infer.hpp>

#include "face_detection.hpp"


namespace p10::recog {
TEST_CASE("recog::FaceDetection::blaze_face", "[recog][face][blaze_face]") {
    IFaceDetector* detector = IFaceDetector::create(
        FaceDetectorConfig::BlazeFace,
        infer::Infer::from_onnx("tests/data/face_detectors/blaze_face.onnx")
            .expect("should load model")
    );

    auto image = io::load_image("tests/data/face_detectors/faces.jpg");
    std::array<FaceDetection, 1> detections;
    detector->detect(image, detections).expect("Face detected");

    EXPECT(detections.faces.size() == 4);
}

}  // namespace p10::recog

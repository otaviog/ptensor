#include <catch2/catch_test_macros.hpp>
#include <ptensor/infer/infer.hpp>
#include <ptensor/infer/infer_config.hpp>
#include <ptensor/io/image.hpp>
#include <ptensor/op/image_layout.hpp>

#include "face_detection.hpp"

namespace p10::recog {
TEST_CASE("recog::FaceDetection::blaze_face", "[recog][face][blaze_face]") {
    IFaceDetector* detector = IFaceDetector::create(
                                  BlazeFaceModel(),
                                  infer::IInfer::from_onnx(
                                      "tests/data/face_detectors/blazefaces-320.onnx",
                                      infer::InferConfig()
                                  )
                                      .expect("should load model")
    )
                                  .expect("Cant create detector");

    Tensor input_tensor;
    op::image_to_tensor(
        io::load_image("tests/data/face_detectors/faces.jpg").expect("should load image"),
        input_tensor,
        Dtype::Uint8,
        op::ImageToTensorNormalize::KeepValues,
        op::ImageToTensorSqueeze::Unsqueze
    );

    std::array<FaceDetection, 1> detections;
    detector->detect(input_tensor, detections).expect("Face detected");
    REQUIRE(detections[0].faces.size() == 4);
}

}  // namespace p10::recog

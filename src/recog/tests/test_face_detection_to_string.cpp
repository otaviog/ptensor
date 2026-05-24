#include <catch2/catch_test_macros.hpp>

#include "face_detection.hpp"
#include "geom/point2.hpp"
#include "geom/rect2.hpp"

namespace p10::recog {

TEST_CASE("recog::Point2::to_string", "[recog][to_string]") {
    REQUIRE(to_string(Point2i {3, 7}) == "{ x: 3, y: 7 }");
}

TEST_CASE("recog::Rect2::to_string", "[recog][to_string]") {
    REQUIRE(
        to_string(Rect2i {Point2i {0, 0}, Point2i {10, 20}})
        == "{ min: { x: 0, y: 0 }, max: { x: 10, y: 20 } }"
    );
}

TEST_CASE("recog::FaceDetection::to_string empty", "[recog][to_string]") {
    REQUIRE(to_string(FaceDetection {}) == "{ faces: [], confidences: [], landmarks: [] }");
}

TEST_CASE("recog::FaceDetection::to_string populated", "[recog][to_string]") {
    FaceDetection det;
    det.faces.push_back(Rect2i {Point2i {0, 0}, Point2i {10, 10}});
    det.faces.push_back(Rect2i {Point2i {5, 5}, Point2i {15, 15}});
    det.confidences.push_back(0.9f);
    det.confidences.push_back(0.5f);
    det.landmarks.push_back({Point2i {1, 2}, Point2i {3, 4}});
    det.landmarks.push_back({});

    REQUIRE(
        to_string(det)
        == "{ faces: ["
           "{ min: { x: 0, y: 0 }, max: { x: 10, y: 10 } }, "
           "{ min: { x: 5, y: 5 }, max: { x: 15, y: 15 } }"
           "], "
           "confidences: [0.9, 0.5], "
           "landmarks: ["
           "[{ x: 1, y: 2 }, { x: 3, y: 4 }], "
           "[]"
           "] }"
    );
}

}  // namespace p10::recog

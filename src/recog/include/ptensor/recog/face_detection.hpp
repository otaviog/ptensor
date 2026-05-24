#pragma once
#include <vector>

#include <ptensor/p10_result.hpp>

#include "geom/point2.hpp"
#include "geom/rect2.hpp"


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

std::string to_string(const FaceDetection &detection);

}  // namespace p10::recog

#pragma once

#include <vector>

#include "React2D.hpp"
#include "Point2D.hpp"

namespace p10::infer::recog {
class FaceDetection {
    std::vector<Rect2D> faces_;
    std::vector<std::vector<Point2D>> coords;
    std::vector<int64_t> labels;
    
};

class IFaceDetector {
public:
    
private:
};

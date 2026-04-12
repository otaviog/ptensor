#pragma once

namespace p10::recog {
class AnchorParameters {
};
class Anchors {
public:
    static Anchors from_parameters(const AnchorParameters &params) {
        
    }
    
private:
    std::vector<Rect2f> anchors;
    Point2f variance;
};
}

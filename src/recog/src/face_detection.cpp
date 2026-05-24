#include "face_detection.hpp"

#include <format>
#include <string>

namespace p10::recog {

std::string to_string(const FaceDetection& detection) {
    std::string faces = "[";
    for (size_t i = 0; i < detection.faces.size(); i++) {
        if (i > 0) {
            faces += ", ";
        }
        faces += to_string(detection.faces[i]);
    }
    faces += "]";

    std::string confidences = "[";
    for (size_t i = 0; i < detection.confidences.size(); i++) {
        if (i > 0) {
            confidences += ", ";
        }
        confidences += std::format("{}", detection.confidences[i]);
    }
    confidences += "]";

    std::string landmarks = "[";
    for (size_t i = 0; i < detection.landmarks.size(); i++) {
        if (i > 0) {
            landmarks += ", ";
        }
        landmarks += "[";
        for (size_t j = 0; j < detection.landmarks[i].size(); j++) {
            if (j > 0) {
                landmarks += ", ";
            }
            landmarks += to_string(detection.landmarks[i][j]);
        }
        landmarks += "]";
    }
    landmarks += "]";

    return std::format(
        "{{ faces: {}, confidences: {}, landmarks: {} }}",
        faces,
        confidences,
        landmarks
    );
}

}  // namespace p10::recog

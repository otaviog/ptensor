#pragma once

namespace p10::infer {

class FaceDetLite: public IFaceDetector {
public:
    P10Error detect(Tensor& images, std::span<FaceDetection> out_detections) override;
    
private:
    std::unique_ptr<infer::IInfer> infer_;
};
}

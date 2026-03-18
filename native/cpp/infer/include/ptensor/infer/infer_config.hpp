#pragma once

namespace p10::infer {
class InferConfig {
  public:
    enum Engine { Onnx };

    InferConfig& engine(Engine engine) {
        engine_ = engine;
        return *this;
    }

    Engine engine() const {
        return engine_;
    }

  private:
    Engine engine_;
};
}  // namespace p10::infer

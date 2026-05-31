#pragma once

namespace p10::infer {
class InferConfig {
  public:
    enum class CoreMLComputeUnits { All, CPUAndGPU, CPUOnly, CPUAndNeuralEngine };

    InferConfig& coreml_compute_units(CoreMLComputeUnits units) {
        coreml_compute_units_ = units;
        return *this;
    }

    CoreMLComputeUnits coreml_compute_units() const {
        return coreml_compute_units_;
    }

  private:
    CoreMLComputeUnits coreml_compute_units_ = CoreMLComputeUnits::All;
};
}  // namespace p10::infer

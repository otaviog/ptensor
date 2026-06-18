#pragma once

#include <memory>
#include <string>
#include <vector>

#include "infer.hpp"
#include "infer_config.hpp"

namespace p10::infer {

// Opaque wrapper around MLModel* — defined in coreml_infer.mm.
struct CoreMLModel;

class CoreMLInfer: public IInfer {
  public:
    static P10Result<std::unique_ptr<IInfer>>
    create(const std::string& model_path, const InferConfig& config);

    ~CoreMLInfer() override;

    P10Error infer(std::span<Tensor> input_tensors, std::span<Tensor> output_tensors) override;

    size_t get_input_count() const override {
        return input_names_.size();
    }

    size_t get_output_count() const override {
        return output_names_.size();
    }

  private:
    CoreMLInfer(
        std::unique_ptr<CoreMLModel> model,
        std::vector<std::string> input_names,
        std::vector<std::string> output_names
    );

    std::unique_ptr<CoreMLModel> model_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};

}  // namespace p10::infer

#pragma once

#include <memory>
#include <span>
#include <string>

#include <ptensor/p10_result.hpp>

namespace p10 {
class Tensor;
}

namespace p10::infer {
class InferConfig;

class IInfer {
  public:
    static P10Result<std::unique_ptr<IInfer>>
    from_onnx(const std::string& onnx_model_path, const InferConfig& config);

    static P10Result<std::unique_ptr<IInfer>>
    from_coreml(const std::string& coreml_model_path, const InferConfig& config);

    IInfer() = default;
    virtual ~IInfer() = default;

    IInfer(const IInfer&) = delete;
    IInfer(const IInfer&&) = delete;
    IInfer& operator=(const IInfer&) = delete;
    IInfer& operator=(const IInfer&&) = delete;

    virtual P10Error infer(std::span<Tensor> input_tensors, std::span<Tensor> output_tensors) = 0;

    virtual size_t get_input_count() const = 0;

    virtual size_t get_output_count() const = 0;
};
}  // namespace p10::infer

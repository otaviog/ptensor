#include <onnxruntime_cxx_api.h>

#include "infer.hpp"

namespace p10::infer {
class OrtInfer: public IInfer {
  public:
    static P10Result<IInfer*> create(const std::string& onnx_path);
    P10Error infer(std::span<Tensor> inputTensors, std::span<Tensor> outputTensors) override;

    size_t get_input_count() const override {
        return session_->GetInputCount();
    }

    size_t get_output_count() const override {
        return session_->GetOutputCount();
    }

  private:
    OrtInfer(const std::string& onnx_model_path, Ort::Env&& env);
    void collect_input_output_names();

    std::string model_path_;
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;

    std::vector<Ort::Value> input_ort_tensors_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::vector<Ort::AllocatedStringPtr> input_names_;
    std::vector<const char*> input_names_cstr_;
    std::vector<Ort::AllocatedStringPtr> output_names_;
    std::vector<const char*> output_names_cstr_;
};
}  // namespace p10::infer

#include "infer.hpp"

#include "infer_config.hpp"
#include "ort_infer.hpp"
#include "ptensor/p10_error.hpp"

namespace p10::infer {
P10Result<IInfer*>
IInfer::from_onnx(const std::string& onnx_model_path, const InferConfig& config) {
    if (config.engine() == InferConfig::Engine::Onnx) {
        return OrtInfer::create(onnx_model_path);
    }
    return Err(P10Error::InvalidArgument << "Unsupported inference engine");
}
}  // namespace p10::infer

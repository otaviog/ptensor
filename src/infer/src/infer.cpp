#include "infer.hpp"

#include "infer_config.hpp"
#include "model_downloader.hpp"
#include "ort_infer.hpp"
#include "ptensor/p10_error.hpp"

namespace p10::infer {
P10Result<std::unique_ptr<IInfer>>
IInfer::from_onnx(const std::string& onnx_model_path, const InferConfig& config) {
    auto local_path_result = resolve_model_path(onnx_model_path);
    if (local_path_result.is_error()) {
        return Err(local_path_result.unwrap_err());
    }
    const std::string local_path = local_path_result.unwrap();

    if (config.engine() == InferConfig::Engine::Ort) {
        return OrtInfer::create(local_path);
    }
    return Err(P10Error::InvalidArgument << "Unsupported inference engine");
}
}  // namespace p10::infer

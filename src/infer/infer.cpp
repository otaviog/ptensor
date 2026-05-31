#include "infer.hpp"

#include "infer_config.hpp"
#include "model_downloader.hpp"
#include "ort_infer.hpp"
#include "ptensor/p10_error.hpp"

#if defined(__APPLE__)
    #include "coreml_infer.hpp"
#endif

namespace p10::infer {
P10Result<std::unique_ptr<IInfer>>
IInfer::from_onnx(const std::string& onnx_model_path, const InferConfig& config) {
    (void)config;
    auto local_path_result = resolve_model_path(onnx_model_path);
    if (local_path_result.is_error()) {
        return Err(local_path_result.unwrap_err());
    }
    return OrtInfer::create(local_path_result.unwrap());
}

P10Result<std::unique_ptr<IInfer>>
IInfer::from_coreml(const std::string& coreml_model_path, const InferConfig& config) {
#if defined(__APPLE__)
    return CoreMLInfer::create(coreml_model_path, config);
#else
    (void)coreml_model_path;
    (void)config;
    return Err(P10Error::NotImplemented << "CoreML is only available on Apple platforms");
#endif
}
}  // namespace p10::infer

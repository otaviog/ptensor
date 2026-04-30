#include "ort_infer.hpp"

#include <cstdint>
#include <memory>

#include <ng-log/logging.h>
#if defined(_MSC_VER)
    #include <ptensor/detail/string.hpp>
#endif
#include <ptensor/tensor.hpp>

#include "ort_conversions.hpp"

namespace p10::infer {
P10Result<IInfer*> OrtInfer::create(const std::string& onnx_path) {
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "hp3d");
        return Ok<IInfer*>(new OrtInfer(onnx_path, std::move(env)));
    } catch (const Ort::Exception& e) {
        return Err(P10Error::InferError << e.what());
    }
}

OrtInfer::OrtInfer(const std::string& modelPath, Ort::Env&& env) :
    model_path_(modelPath),
    env_(std::move(env)),
    session_options_() {
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    session_.reset(
        new Ort::Session(
            env_,
#if defined(_MSC_VER)
            string_to_wstring(model_path_).unwrap().c_str(),
#else
            model_path_.c_str(),
#endif
            session_options_
            )
    );

    collect_input_output_names();
}

void OrtInfer::collect_input_output_names() {
    // Get input/output names

    for (size_t i = 0; i < session_->GetInputCount(); ++i) {
        auto input_name = session_->GetInputNameAllocated(i, allocator_);
        input_names_cstr_.emplace_back(input_name.get());
        input_names_.emplace_back(std::move(input_name));
    }

    for (size_t i = 0; i < session_->GetOutputCount(); i++) {
        auto output_name = session_->GetOutputNameAllocated(i, allocator_);
        output_names_cstr_.emplace_back(output_name.get());
        output_names_.emplace_back(std::move(output_name));
    }

    LOG(INFO) << "ONNX model loaded: " << model_path_;
    LOG(INFO) << "Number of input: " << input_names_cstr_.size();
    LOG(INFO) << "Number of outputs: " << output_names_cstr_.size();
}

P10Error OrtInfer::infer(std::span<Tensor> input_tensors, std::span<Tensor> output_tensors) {
    try {
        if (input_tensors.size() != get_input_count()
            || output_tensors.size() != get_output_count()) {
            return P10Error::InvalidArgument << "Invalid number of input/output tensors";
        }

        // Setup memory info for tensors
        Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::AllocatorWithDefaultOptions allocator;

        input_ort_tensors_.clear();
        // Create input tensors if not already created
        for (auto& tensor : input_tensors) {
            auto ort_value = Ort::Value::CreateTensor(
                memory_info,
                tensor.as_bytes().data(),
                tensor.size_bytes(),
                tensor.shape().as_span().data(),
                tensor.shape().dims(),
                dtype_to_ort(tensor.dtype())
            );
            input_ort_tensors_.emplace_back(std::move(ort_value));
        }

        // Run inference
        auto result = session_->Run(
            Ort::RunOptions {nullptr},
            input_names_cstr_.data(),
            input_ort_tensors_.data(),
            input_ort_tensors_.size(),
            output_names_cstr_.data(),
            output_names_cstr_.size()
        );

        for (size_t i = 0; i < result.size(); i++) {
            const auto& result_ort_value = result[i];

            const auto& type_and_shape_info = result_ort_value.GetTensorTypeAndShapeInfo();
            const auto output_shape = type_and_shape_info.GetShape();
            const auto output_data = result_ort_value.GetTensorData<uint8_t>();

            auto err = ort_to_hp3d_dtype(type_and_shape_info.GetElementType());
            if (!err.is_ok()) {
                return err.error();
            }

            auto dtype = err.unwrap();
            auto shape_result = make_shape(std::span<const int64_t>(output_shape));
            if (!shape_result.is_ok()) {
                return shape_result.error();
            }
            P10_RETURN_IF_ERROR(
                output_tensors[i].create(shape_result.unwrap(), TensorOptions().dtype(dtype))
            );
            std::memcpy(
                output_tensors[i].as_bytes().data(),
                output_data,
                output_tensors[i].size_bytes()
            );
        }

        return P10Error::Ok;
    } catch (const Ort::Exception& e) {
        return P10Error::InferError << e.what();
    }
}
}  // namespace p10::infer

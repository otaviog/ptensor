#include "ptensor_infer.h"

#include <vector>

#include <ptensor/infer/infer.hpp>
#include <ptensor/infer/infer_config.hpp>
#include <ptensor/tensor.hpp>

#include "infer_wrapper.hpp"
#include "tensor_wrapper.hpp"
#include "update_error_state.hpp"

PTENSOR_API P10ErrorEnum p10_infer_from_onnx(P10Infer* infer, const char* onnx_model_path) {
    auto result = p10::infer::IInfer::from_onnx(onnx_model_path, p10::infer::InferConfig());
    if (result.is_error()) {
        return p10::update_error_state(result.unwrap_err());
    }
    *infer = p10::infer::wrap_infer(result.unwrap());
    return P10_OK;
}

PTENSOR_API P10ErrorEnum p10_infer_destroy(P10Infer* infer) {
    if (infer == nullptr || *infer == nullptr) {
        return P10_OK;
    }
    delete p10::infer::unwrap_infer(*infer);
    *infer = nullptr;
    return P10_OK;
}

PTENSOR_API size_t p10_infer_get_input_count(P10Infer infer) {
    return p10::infer::unwrap_infer(infer)->get_input_count();
}

PTENSOR_API size_t p10_infer_get_output_count(P10Infer infer) {
    return p10::infer::unwrap_infer(infer)->get_output_count();
}

PTENSOR_API P10ErrorEnum p10_infer_run(
    P10Infer infer,
    const Ptensor* input_tensors,
    size_t num_inputs,
    Ptensor* output_tensors,
    size_t num_outputs
) {
    // Build non-owning input views so the caller's tensors are not consumed.
    std::vector<p10::Tensor> cpp_inputs;
    cpp_inputs.reserve(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
        const p10::Tensor* src = p10::unwrap(input_tensors[i]);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        void* data = const_cast<void*>(static_cast<const void*>(src->as_bytes().data()));
        cpp_inputs.push_back(
            p10::Tensor::from_data(
                data,
                src->shape(),
                p10::TensorOptions().dtype(src->dtype()).stride(src->stride())
            )
        );
    }

    // Output slots: default-constructed Tensors that infer() will fill via create().
    std::vector<p10::Tensor> cpp_outputs(num_outputs);

    auto err = p10::infer::unwrap_infer(infer)->infer(cpp_inputs, cpp_outputs);
    if (err.is_error()) {
        return p10::update_error_state(err);
    }

    for (size_t i = 0; i < num_outputs; ++i) {
        output_tensors[i] = p10::wrap(new p10::Tensor(std::move(cpp_outputs[i])));
    }

    return P10_OK;
}

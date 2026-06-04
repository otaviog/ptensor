#include "blaze_face.hpp"

#include <ptensor/infer/infer.hpp>
#include <p10_internal/log/log.hpp>
#include <ptensor/op/resize.hpp>
#include <ptensor/op/statistics.hpp>
#include <ptensor/tensor.hpp>

namespace p10::recog {

P10Error BlazeFace::detect(Tensor& images, std::span<FaceDetection> out_detections) {
    P10_RETURN_IF_ERROR(verify_detect_arguments(images, out_detections));

    auto preproc_result = pre_process_.process(images, input_buffer_[0]);
    if (preproc_result.is_error()) {
        return preproc_result.error();
    }
    const float resize_ratio = preproc_result.unwrap();
    P10_RETURN_IF_ERROR(infer_->infer(input_buffer_, outputs_));

    // Anchors are normalized to the *padded resized* image the model actually
    // saw, not the original. Read those dimensions back from the preprocessed
    // buffer ([N, C, H, W]).
    const auto processed_shape = input_buffer_[0].shape().as_span();
    const auto processed_height = static_cast<size_t>(processed_shape[2]);
    const auto processed_width = static_cast<size_t>(processed_shape[3]);
    post_process_
        .process(processed_width, processed_height, resize_ratio, outputs_, out_detections);

    return P10Error::Ok;
}

}  // namespace p10::recog

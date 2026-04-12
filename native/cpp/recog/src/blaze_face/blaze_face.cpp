#include "blaze_face.hpp"

#include <ptensor/op/resize.hpp>
#include <ptensor/tensor.hpp>

#include "ptensor/p10_error.hpp"

namespace p10::recog {

P10Error BlazeFace::detect(Tensor& images, std::span<FaceDetection> out_detections) {
    const auto input_shape = images.shape().as_span();
    if (input_shape.size() != 4) {
        return P10Error::InvalidArgument << "Input tensor must have 4 dimensions [N x C x H x W]";
    }
    if (input_shape[1] != 3) {
        return P10Error::InvalidArgument << "Input tensor must have 3 channels (C=3)";
    }

    if (input_shape[0] != out_detections.size()) {
        return P10Error::InvalidArgument
            << "Output detections size must match the number of input images";
    }

    const auto input_width = input_shape[3];
    const auto input_height = input_shape[2];
    auto preproc_result = pre_process_.process(images, input_buffer_[0]);
    if (preproc_result.is_error()) {
        return preproc_result.error();
    }

    P10_RETURN_IF_ERROR(infer_->infer(input_buffer_, outputs_));
    post_process_
        .process(input_width, input_height, preproc_result.unwrap(), outputs_, out_detections);

    return P10Error::Ok;
}

}  // namespace p10::recog

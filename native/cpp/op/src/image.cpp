#include "image.hpp"

#include <algorithm>

#include <ptensor/tensor.hpp>

namespace p10::op {
P10Error image_to_tensor(const Tensor& image, Tensor& tensor) {
    if (image.dtype() != Dtype::Uint8) {
        return P10Error::InvalidArgument << "Input tensor must be of type UINT8.";
    }
    if (image.shape().dims() != 3) {
        return P10Error::InvalidArgument
            << "Input tensor must have shape [height, width, channels].";
    }
    if (!image.is_contiguous()) {
        return P10Error::InvalidArgument << "Input tensor must be contiguous in memory.";
    }

    const auto image_span = image.as_span3d<uint8_t>().expect("Invalid image");

    const auto num_channels = image_span.channels();
    tensor.create(
        make_shape(
            int64_t(num_channels),
            int64_t(image_span.height()),
            int64_t(image_span.width())
        ),
        Dtype::Float32
    );
    auto tensor_span = tensor.as_planar_span3d<float>().unwrap();

    for (size_t row = 0; row < image_span.height(); row++) {
        for (size_t col = 0; col < image_span.width(); col++) {
            const auto& input_channel = image_span.channel(row, col);
            for (size_t c = 0; c < num_channels; c++) {
                tensor_span[c].row(row)[col] = float(input_channel[c]) / 255.0f;
            }
        }
    }

    return P10Error::Ok;
}

P10Error image_from_tensor(const Tensor& tensor, Tensor& image) {
    if (tensor.dtype() != Dtype::Float32) {
        throw P10Error::InvalidArgument << "Input tensor must be of type FLOAT32.";
    }
    if (tensor.shape().dims() != 3) {
        throw P10Error::InvalidArgument
            << "Input tensor must have shape [channels, height, width].";
    }
    // if (!tensor.is_contiguous()) {
    //     throw P10Error::InvalidArgument << "Input tensor must be contiguous in memory.";
    // }

    const auto num_planes = size_t(tensor.shape(0).unwrap());
    const auto height = size_t(tensor.shape(1).unwrap());
    const auto width = size_t(tensor.shape(2).unwrap());

    const auto input_span = tensor.as_planar_span3d<float>().unwrap();
    image.create(make_shape(int64_t(height), int64_t(width), int64_t(num_planes)), Dtype::Uint8);

    auto output_span = image.as_span3d<uint8_t>().unwrap();

    for (size_t row = 0; row < input_span.height(); row++) {
        for (size_t col = 0; col < input_span.width(); col++) {
            auto output_channel = output_span.channel(row, col);

            for (size_t c = 0; c < num_planes; c++) {
                float value = input_span[c].row(row)[col] * 255.0f;
                value = std::clamp(value, 0.0f, 255.0f);
                output_channel[c] = static_cast<uint8_t>(value);
            }
        }
    }
    return P10Error::Ok;
}
}  // namespace p10::op

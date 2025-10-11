#include "image.hpp"

#include <algorithm>
#include <array>

#include <ptensor/tensor.hpp>

namespace p10::op {
PtensorError image_to_tensor(const Tensor& image, Tensor& tensor) {
    if (image.dtype() != Dtype::Uint8) {
        return PtensorError::InvalidArgument << "Input tensor must be of type UINT8.";
    }
    if (image.shape().dims() != 3) {
        return PtensorError::InvalidArgument
            << "Input tensor must have shape [height, width, channels].";
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

    std::array<Span2D<float>, P10_MAX_SHAPE> planes_array;
    auto planes = std::span(planes_array.data(), num_channels);
    for (size_t c = 0; c < num_channels; c++) {
        planes[c] = tensor_span.plane(c);
    }

    for (size_t row = 0; row < image_span.height(); row++) {
        for (size_t col = 0; col < image_span.width(); col++) {
            const auto& input_channel = image_span.channel(row, col);
            const size_t plannar_offset = row * image_span.width() + col;
            for (size_t c = 0; c < num_channels; c++) {
                planes[c][plannar_offset] = float(input_channel[c]) / 255.0f;
            }
        }
    }

    return PtensorError::Ok;
}

PtensorError image_from_tensor(const Tensor& tensor, Tensor& image) {
    if (tensor.dtype() != Dtype::Float32) {
        throw PtensorError::InvalidArgument << "Input tensor must be of type FLOAT32.";
    }
    if (tensor.shape().dims() != 3) {
        throw PtensorError::InvalidArgument
            << "Input tensor must have shape [channels, height, width].";
    }

    const auto numPlanes = size_t(tensor.shape(0).unwrap());
    const auto height = size_t(tensor.shape(1).unwrap());
    const auto width = size_t(tensor.shape(2).unwrap());

    const auto input_span = tensor.as_planar_span3d<float>().unwrap();
    image.create(make_shape(int64_t(height), int64_t(width), int64_t(numPlanes)), Dtype::Uint8);

    auto output_span = image.as_span3d<uint8_t>().unwrap();

    std::array<Span2D<const float>, P10_MAX_SHAPE> planes_array;
    auto planes = std::span(planes_array.data(), numPlanes);
    for (size_t p = 0; p < numPlanes; p++) {
        planes[p] = input_span.plane(p);
    }

    for (size_t row = 0; row < input_span.height(); row++) {
        for (size_t col = 0; col < input_span.width(); col++) {
            auto output_channel = output_span.channel(row, col);

            for (size_t c = 0; c < 3; c++) {
                float value = planes[c].row(row)[col] * 255.0f;
                value = std::clamp(value, 0.0f, 255.0f);
                output_channel[c] = static_cast<uint8_t>(value);
            }
        }
    }
    return PtensorError::Ok;
}
}  // namespace p10::op

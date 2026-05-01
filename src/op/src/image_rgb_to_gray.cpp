#include "image_rgb_to_gray.hpp"

#include <ptensor/tensor.hpp>

#include "ptensor/p10_error.hpp"

namespace p10::op {
p10::P10Error image_rgb_to_gray(const p10::Tensor& rgb_image, p10::Tensor& gray_image) {
    if (rgb_image.dims() != 3 || rgb_image.shape(2).unwrap() != 3) {
        return P10Error::InvalidArgument << "Input RGB image must have 3 dimensions [H x W x 3]";
    }

    if (rgb_image.dtype() != Dtype::Uint8) {
        return P10Error::InvalidArgument << "Input RGB image must have Uint8 dtype";
    }

    auto height = rgb_image.shape(0).unwrap();
    auto width = rgb_image.shape(1).unwrap();
    P10_RETURN_IF_ERROR(gray_image.create(make_shape(height, width), Dtype::Uint8));

    auto rgb_span = rgb_image.as_span3d<const uint8_t>().unwrap();
    auto gray_span = gray_image.as_span2d<uint8_t>().unwrap();

    for (size_t h = 0; h < static_cast<size_t>(height); ++h) {
        const auto rgb_row = rgb_span.row(h);
        auto gray_row = gray_span.row(h);
        for (size_t w = 0; w < static_cast<size_t>(width); ++w) {
            auto rgb_pixel = &rgb_row[w * 3];
            uint8_t r = rgb_pixel[0];
            uint8_t g = rgb_pixel[1];
            uint8_t b = rgb_pixel[2];

            // Convert to grayscale using luminosity method
            gray_row[w] = static_cast<uint8_t>(
                0.21f * static_cast<float>(r) + 0.72f * static_cast<float>(g)
                + 0.07f * static_cast<float>(b)
            );
        }
    }

    return P10Error::Ok;
}
}  // namespace p10::op

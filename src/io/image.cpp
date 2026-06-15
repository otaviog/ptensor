#include "image.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace p10::io {

P10Result<Tensor> load_image(const std::string& path) {
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 0);
    if (data == nullptr) {
        return Err(P10Error::InvalidArgument, "Failed to load image");
    }
    Tensor tensor;
    tensor.create(make_shape(height, width, channels), Dtype::Uint8);
    auto tensor_bytes = tensor.as_bytes();
    std::ranges::copy(
        std::span(reinterpret_cast<std::byte*>(data), tensor_bytes.size()),
        tensor_bytes.data()
    );

    stbi_image_free(data);
    return Ok(std::move(tensor));
}

P10Error save_image(const std::string& path, const Tensor& tensor) {
    if (tensor.dtype() != Dtype::Uint8) {
        return P10Error::InvalidArgument << "Only uint8 tensors can be saved as images";
    }

    if (tensor.dims() == 2) {
        const auto span = tensor.as_span2d<uint8_t>().unwrap();
        stbi_write_png(
            path.c_str(),
            static_cast<int>(span.cols()),
            static_cast<int>(span.rows()),
            1,
            span[0].data(),
            static_cast<int>(span.cols())
        );
    } else if (tensor.dims() == 3) {
        // HWC: shape is (height, width, channels).
        const auto height = static_cast<int>(tensor.shape(0).unwrap());
        const auto width = static_cast<int>(tensor.shape(1).unwrap());
        const auto channels = static_cast<int>(tensor.shape(2).unwrap());
        stbi_write_png(
            path.c_str(),
            width,
            height,
            channels,
            tensor.as_bytes().data(),
            width * channels
        );
    }
    return P10Error::Ok;
}

}  // namespace p10::io

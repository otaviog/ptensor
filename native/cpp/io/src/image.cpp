#include "image.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace p10::io {

P10Result<Tensor> load_image(const std::string& path) {
    int width, height, channels;
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 0);
    if (!data) {
        return Err(P10Error::InvalidArgument, "Failed to load image");
    }
    Tensor tensor;
    tensor.create(make_shape(height, width, channels), Dtype::Uint8);
    auto tensor_bytes = tensor.as_bytes();
    std::copy(
        data,
        data + tensor_bytes.size(),
        reinterpret_cast<unsigned char*>(tensor_bytes.data())
    );
    stbi_image_free(data);
    return Ok<Tensor>(std::move(tensor));
}

P10Error save_image(const std::string& path, const Tensor& tensor) {
    if (tensor.dtype() != Dtype::Uint8) {
        return P10Error::InvalidArgument << "Only uint8 tensors can be saved as images";
    }

    if (tensor.dims() == 2) {
        const auto span = tensor.as_span2d<uint8_t>().unwrap();
        stbi_write_png(
            path.c_str(),
            int(span.width()),
            int(span.height()),
            1,
            span.row(0),
            int(span.width())
        );
    } else if (tensor.dims() == 3) {
        auto span = tensor.as_span3d<uint8_t>().unwrap();
        stbi_write_png(
            path.c_str(),
            int(span.width()),
            int(span.height()),
            int(span.channels()),
            span.data(),
            int(span.width() * span.channels())
        );
    }
    return P10Error::Ok;
}

}  // namespace p10::io

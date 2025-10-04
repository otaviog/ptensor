#include "image.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace p10::io {

PtensorResult<Tensor> load_image(const std::string& path) {
    int width, height, channels;
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 0);
    if (!data) {
        return Err(PtensorError::InvalidArgument, "Failed to load image");
    }
    Tensor tensor;
    tensor.create(make_shape({height, width, channels}).unwrap(), Dtype::Uint8);
    auto tensor_bytes = tensor.as_bytes();
    std::copy(
        data,
        data + tensor_bytes.size(),
        reinterpret_cast<unsigned char*>(tensor_bytes.data())
    );
    stbi_image_free(data);
    return Ok<Tensor>(std::move(tensor));
}

PtensorError save_image(const std::string& path, const Tensor& tensor) {
    auto span = tensor.as_span3d<uint8_t>().unwrap();
    stbi_write_png(
        path.c_str(),
        span.width(),
        span.height(),
        span.channels(),
        span.data(),
        span.width() * span.channels()
    );
    return PtensorError::Ok;
}

}  // namespace p10::io

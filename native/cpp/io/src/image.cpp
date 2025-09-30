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
        return Err<Tensor>(PtensorError::INVALID_ARGUMENT, "Failed to load image");
    }
    Tensor tensor;
    tensor.create(DType::UINT8, {height, width, channels});
    std::copy(data, data + height * width * channels, tensor.data<uint8_t>());
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
    return PtensorError::OK;
}

}  // namespace p10::io

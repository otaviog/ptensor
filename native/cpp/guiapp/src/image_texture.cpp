#include "image_texture.hpp"

#include <cstring>

#include <vulkan/vulkan.h>

#include "image_texture.impl.hpp"
#include "ptensor/guiapp/gui_app.hpp"

namespace p10::guiapp {

ImageTexture::ImageTexture() : impl_(nullptr) {}

ImageTexture::ImageTexture(Impl* impl) : impl_(impl) {}

ImageTexture::~ImageTexture() = default;

ImageTexture::ImageTexture(ImageTexture&&) noexcept = default;
ImageTexture& ImageTexture::operator=(ImageTexture&&) noexcept = default;

P10Error ImageTexture::upload(const Tensor& tensor) {
    if (!impl_) {
        return P10Error::InvalidOperation << "ImageTexture not initialized";
    }
    return impl_->upload(tensor);
}

ImTextureID ImageTexture::texture_id() const {
    if (!impl_) {
        return (ImTextureID)0;
    }
    return impl_->texture_id();
}

int ImageTexture::width() const {
    if (!impl_) {
        return 0;
    }
    return impl_->width();
}

int ImageTexture::height() const {
    if (!impl_) {
        return 0;
    }
    return impl_->height();
}

bool ImageTexture::is_valid() const {
    if (!impl_) {
        return false;
    }
    return impl_->is_valid();
}

void ImageTexture::clear() {
    if (impl_) {
        impl_->clear();
    }
}

}  // namespace p10::guiapp

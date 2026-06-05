#include "image_texture.impl.hpp"

#include <Metal/Metal.hpp>
#include <ptensor/tensor.hpp>

namespace p10::viz {

P10Error ImageTexture::Impl::upload(const Tensor& tensor, TensorLayout layout) {
    auto staged = stager_.stage(tensor, layout);
    if (staged.is_error()) {
        return staged.error();
    }
    const UploadView view = staged.unwrap();

    if (!is_valid() || width_ != view.width || height_ != view.height || tex_fmt_ != view.format) {
        tex_fmt_ = view.format;
        return create_texture(view.width, view.height, view.data);
    }
    return upload_data(view.data);
}

P10Error ImageTexture::Impl::create_texture(int width, int height, const void* data) {
    clear();

    auto* device = reinterpret_cast<MTL::Device*>(ctx_.device);

    MTL::TextureDescriptor* desc = MTL::TextureDescriptor::alloc()->init();
    desc->setTextureType(MTL::TextureType2D);
    desc->setWidth(static_cast<NS::UInteger>(width));
    desc->setHeight(static_cast<NS::UInteger>(height));
    desc->setMipmapLevelCount(1);
    desc->setUsage(MTL::TextureUsageShaderRead);
    desc->setStorageMode(MTL::StorageModeManaged);

    if (tex_fmt_ == TextureFormat::Gray8) {
        desc->setPixelFormat(MTL::PixelFormatR8Unorm);
        desc->setSwizzle(
            MTL::TextureSwizzleChannels::Make(
                MTL::TextureSwizzleRed,
                MTL::TextureSwizzleRed,
                MTL::TextureSwizzleRed,
                MTL::TextureSwizzleOne
            )
        );
    } else {
        desc->setPixelFormat(MTL::PixelFormatRGBA8Unorm);
    }

    MTL::Texture* texture = device->newTexture(desc);
    desc->release();

    if (texture == nullptr) {
        return P10Error::InvalidOperation << "Failed to create Metal texture";
    }

    width_ = width;
    height_ = height;
    texture_ = texture;

    return upload_data(data);
}

P10Error ImageTexture::Impl::upload_data(const void* data) {
    auto* texture = reinterpret_cast<MTL::Texture*>(texture_);
    const NS::UInteger bytes_per_row = (tex_fmt_ == TextureFormat::Gray8)
        ? static_cast<NS::UInteger>(width_)
        : static_cast<NS::UInteger>(width_) * 4;

    MTL::Region region = MTL::Region::Make2D(
        0,
        0,
        static_cast<NS::UInteger>(width_),
        static_cast<NS::UInteger>(height_)
    );
    texture->replaceRegion(region, 0, data, bytes_per_row);
    return P10Error::Ok;
}

void ImageTexture::Impl::clear() {
    if (texture_ != nullptr) {
        reinterpret_cast<MTL::Texture*>(texture_)->release();
        texture_ = nullptr;
    }
    width_ = 0;
    height_ = 0;
}

}  // namespace p10::viz

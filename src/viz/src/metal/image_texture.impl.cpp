#include "image_texture.impl.hpp"

#include <ptensor/tensor.hpp>

// Metal-CPP headers — private implementations are defined in gui_app.impl.cpp.
// Include here for declarations only.
#include <Metal/Metal.hpp>

namespace p10::viz {

P10Error ImageTexture::Impl::upload(const Tensor& tensor) {
    if (tensor.dims() != 3) {
        return P10Error::InvalidArgument
            << ("Tensor must have rank 3 (H, W, C), got rank " + std::to_string(tensor.dims()));
    }

    const int64_t height = tensor.shape()[0].unwrap();
    const int64_t width = tensor.shape()[1].unwrap();
    const int64_t channels = tensor.shape()[2].unwrap();

    if (channels != 1 && channels != 3 && channels != 4) {
        return P10Error::InvalidArgument
            << ("Channels must be 1, 3, or 4, got " + std::to_string(channels));
    }

    std::vector<uint8_t> rgba_data(static_cast<size_t>(width * height * 4));

    if (tensor.dtype() == Dtype::Uint8) {
        auto src_result = tensor.as_span1d<uint8_t>();
        if (src_result.is_error())
            return src_result.error();
        auto src = src_result.unwrap();

        for (int64_t i = 0; i < height * width; ++i) {
            if (channels == 1) {
                rgba_data[i * 4 + 0] = src[i];
                rgba_data[i * 4 + 1] = src[i];
                rgba_data[i * 4 + 2] = src[i];
                rgba_data[i * 4 + 3] = 255;
            } else if (channels == 3) {
                rgba_data[i * 4 + 0] = src[i * 3 + 0];
                rgba_data[i * 4 + 1] = src[i * 3 + 1];
                rgba_data[i * 4 + 2] = src[i * 3 + 2];
                rgba_data[i * 4 + 3] = 255;
            } else {
                rgba_data[i * 4 + 0] = src[i * 4 + 0];
                rgba_data[i * 4 + 1] = src[i * 4 + 1];
                rgba_data[i * 4 + 2] = src[i * 4 + 2];
                rgba_data[i * 4 + 3] = src[i * 4 + 3];
            }
        }
    } else if (tensor.dtype() == Dtype::Float32) {
        auto src_result = tensor.as_span1d<float>();
        if (src_result.is_error())
            return src_result.error();
        auto src = src_result.unwrap();

        for (int64_t i = 0; i < height * width; ++i) {
            if (channels == 1) {
                uint8_t val = static_cast<uint8_t>(src[i] * 255.0f);
                rgba_data[i * 4 + 0] = val;
                rgba_data[i * 4 + 1] = val;
                rgba_data[i * 4 + 2] = val;
                rgba_data[i * 4 + 3] = 255;
            } else if (channels == 3) {
                rgba_data[i * 4 + 0] = static_cast<uint8_t>(src[i * 3 + 0] * 255.0f);
                rgba_data[i * 4 + 1] = static_cast<uint8_t>(src[i * 3 + 1] * 255.0f);
                rgba_data[i * 4 + 2] = static_cast<uint8_t>(src[i * 3 + 2] * 255.0f);
                rgba_data[i * 4 + 3] = 255;
            } else {
                rgba_data[i * 4 + 0] = static_cast<uint8_t>(src[i * 4 + 0] * 255.0f);
                rgba_data[i * 4 + 1] = static_cast<uint8_t>(src[i * 4 + 1] * 255.0f);
                rgba_data[i * 4 + 2] = static_cast<uint8_t>(src[i * 4 + 2] * 255.0f);
                rgba_data[i * 4 + 3] = static_cast<uint8_t>(src[i * 4 + 3] * 255.0f);
            }
        }
    } else {
        return P10Error::InvalidArgument << "Unsupported dtype: " << to_string(tensor.dtype());
    }

    const int new_width = static_cast<int>(width);
    const int new_height = static_cast<int>(height);

    if (!is_valid() || width_ != new_width || height_ != new_height) {
        return create_texture(new_width, new_height, rgba_data.data());
    } else {
        return upload_data(rgba_data.data());
    }
}

P10Error ImageTexture::Impl::create_texture(int width, int height, const void* data) {
    clear();

    auto* device = reinterpret_cast<MTL::Device*>(ctx_.device);

    MTL::TextureDescriptor* desc = MTL::TextureDescriptor::alloc()->init();
    desc->setTextureType(MTL::TextureType2D);
    desc->setPixelFormat(MTL::PixelFormatRGBA8Unorm);
    desc->setWidth(static_cast<NS::UInteger>(width));
    desc->setHeight(static_cast<NS::UInteger>(height));
    desc->setMipmapLevelCount(1);
    desc->setUsage(MTL::TextureUsageShaderRead);
    desc->setStorageMode(MTL::StorageModeManaged);

    MTL::Texture* texture = device->newTexture(desc);
    desc->release();

    if (!texture) {
        return P10Error::InvalidOperation << "Failed to create Metal texture";
    }

    width_ = width;
    height_ = height;
    texture_ = texture;  // retain count == 1 from newTexture()

    return upload_data(data);
}

P10Error ImageTexture::Impl::upload_data(const void* data) {
    auto* texture = reinterpret_cast<MTL::Texture*>(texture_);
    MTL::Region region =
        MTL::Region::Make2D(0, 0, static_cast<NS::UInteger>(width_), static_cast<NS::UInteger>(height_));
    texture->replaceRegion(region, 0, data, static_cast<NS::UInteger>(width_ * 4));
    return P10Error::Ok;
}

void ImageTexture::Impl::clear() {
    if (texture_) {
        reinterpret_cast<MTL::Texture*>(texture_)->release();
        texture_ = nullptr;
    }
    width_ = 0;
    height_ = 0;
}

}  // namespace p10::viz

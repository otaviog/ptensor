#pragma once

#include <ptensor/p10_error.hpp>

#include "image_texture.hpp"

namespace p10 {
class Tensor;
}

namespace p10::viz {

struct ImageTextureMetalContext {
    void* device;         // id<MTLDevice> — not retained here, owned by GuiApp::Impl
    void* command_queue;  // id<MTLCommandQueue> — not retained here, owned by GuiApp::Impl
};

class ImageTexture::Impl {
  public:
    Impl(ImageTextureMetalContext ctx) : ctx_(ctx) {}

    ~Impl() {
        clear();
    }

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    P10Error upload(const Tensor& tensor);

    ImTextureID texture_id() const {
        return (ImTextureID)(uintptr_t)texture_;
    }

    int width() const {
        return width_;
    }

    int height() const {
        return height_;
    }

    bool is_valid() const {
        return texture_ != nullptr;
    }

    void clear();

  private:
    P10Error create_texture(int width, int height, const void* data);
    P10Error upload_data(const void* data);

    ImageTextureMetalContext ctx_;
    void* texture_ = nullptr;  // CFTypeRef (retained) to id<MTLTexture>
    int width_ = 0;
    int height_ = 0;
};

}  // namespace p10::viz

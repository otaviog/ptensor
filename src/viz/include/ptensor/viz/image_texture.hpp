#pragma once

#include <memory>

#include <imgui.h>
#include <ptensor/p10_error.hpp>
#include <ptensor/tensor.hpp>

namespace p10::viz {

/// GPU texture for rendering with ImGui. Hides backend (Vulkan/Metal) details.
class ImageTexture {
  public:
    class Impl;

    /// Create an uninitialized ImageTexture.
    ImageTexture();

    /// Create an ImageTexture associated with a GuiApp.
    explicit ImageTexture(Impl* impl);

    ~ImageTexture();

    ImageTexture(const ImageTexture&) = delete;
    ImageTexture& operator=(const ImageTexture&) = delete;

    ImageTexture(ImageTexture&&) noexcept;
    ImageTexture& operator=(ImageTexture&&) noexcept;

    /// Upload a tensor to GPU as a texture (shape [H, W, C], C in {1, 3, 4}, dtype UInt8 or Float32).
    P10Error upload(const Tensor& tensor);

    /// Get the ImGui texture ID for use with ImGui::Image().
    ImTextureID texture_id() const;

    /// Get the texture width in pixels.
    int width() const;

    /// Get the texture height in pixels.
    int height() const;

    /// Check if a valid texture has been uploaded.
    bool is_valid() const;

    /// Release GPU resources for this texture.
    void clear();

  private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace p10::viz

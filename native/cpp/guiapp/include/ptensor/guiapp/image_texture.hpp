#pragma once

#include <memory>

#include <imgui.h>
#include <ptensor/p10_error.hpp>
#include <ptensor/tensor.hpp>

namespace p10::guiapp {

/**
 * Manages a Vulkan texture that can be rendered with ImGui.
 * Uses pimpl pattern to hide Vulkan implementation details.
 */
class ImageTexture {
  public:
    class Impl;

    /**
     * Create an uninitialized ImageTexture.
     * Must be assigned from create_texture() before use.
     */
    ImageTexture();

    /**
     * Create an ImageTexture associated with a GuiApp.
     * The GuiApp must remain valid for the lifetime of this ImageTexture.
     */
    explicit ImageTexture(Impl* impl);

    ~ImageTexture();

    ImageTexture(const ImageTexture&) = delete;
    ImageTexture& operator=(const ImageTexture&) = delete;

    ImageTexture(ImageTexture&&) noexcept;
    ImageTexture& operator=(ImageTexture&&) noexcept;

    /**
     * Upload a tensor to GPU as a texture.
     *
     * Supported tensor formats:
     * - Shape: [H, W, C] where C is 1 (grayscale), 3 (RGB), or 4 (RGBA)
     * - Dtype: UInt8, Float32
     *
     * For Float32 tensors, values are expected in [0, 1] range and will be
     * converted to UInt8 internally.
     */
    P10Error upload(const Tensor& tensor);

    /**
     * Get the ImGui texture ID for use with ImGui::Image().
     * Returns nullptr if no texture has been uploaded.
     */
    ImTextureID texture_id() const;

    /**
     * Get the texture width in pixels.
     */
    int width() const;

    /**
     * Get the texture height in pixels.
     */
    int height() const;

    /**
     * Check if a valid texture has been uploaded.
     */
    bool is_valid() const;

    /**
     * Release GPU resources for this texture.
     */
    void clear();

  private:
    std::unique_ptr<Impl> impl_;
};
}  // namespace p10::guiapp

#pragma once

#include <ptensor/p10_error.hpp>
#include <vulkan/vulkan.h>

#include "image_texture.hpp"

namespace p10 {
class Tensor;
}

namespace p10::guiapp {
struct ImageTextureVkContext {
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
};

class ImageTexture::Impl {
  public:
    Impl(ImageTextureVkContext context) : context_(context) {}

    ~Impl() {
        clear();
    }

    P10Error upload(const Tensor& tensor);

    ImTextureID get_texture_id() const {
        return (ImTextureID)descriptor_set_;
    }

    int width() const {
        return width_;
    }

    int height() const {
        return height_;
    }

    bool is_valid() const {
        return descriptor_set_ != VK_NULL_HANDLE;
    }

    void clear();

  private:
    P10Error create_texture(int width, int height, const void* data);
    P10Error create_image(int width, int height);
    P10Error allocate_memory();
    P10Error upload_data(const void* data, size_t size);
    P10Error create_image_view();
    P10Error create_sampler();
    P10Error create_descriptor_set();

    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties);

    int width_ = 0;
    int height_ = 0;

    ImageTextureVkContext context_;

    VkImage image_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkImageView image_view_ = VK_NULL_HANDLE;
    VkSampler sampler_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
};

}  // namespace p10::guiapp
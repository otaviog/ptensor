#include "image_texture.impl.hpp"

#include <ptensor/tensor.hpp>

#include "imgui_impl_vulkan.h"
#include "vulkan_context.hpp"
#include "vulkan_error.hpp"

namespace p10::guiapp {

P10Error ImageTexture::Impl::upload(const Tensor& tensor) {
    // Validate tensor shape
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

    // Prepare pixel data
    std::vector<uint8_t> rgba_data(static_cast<size_t>(width * height * 4));

    if (tensor.dtype() == Dtype::Uint8) {
        auto src_result = tensor.as_span1d<uint8_t>();
        if (src_result.is_error()) {
            return src_result.error();
        }
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
            } else {  // channels == 4
                rgba_data[i * 4 + 0] = src[i * 4 + 0];
                rgba_data[i * 4 + 1] = src[i * 4 + 1];
                rgba_data[i * 4 + 2] = src[i * 4 + 2];
                rgba_data[i * 4 + 3] = src[i * 4 + 3];
            }
        }
    } else if (tensor.dtype() == Dtype::Float32) {
        auto src_result = tensor.as_span1d<float>();
        if (src_result.is_error()) {
            return src_result.error();
        }
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
            } else {  // channels == 4
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

    // Only recreate texture if dimensions changed or texture doesn't exist
    if (!is_valid() || width_ != new_width || height_ != new_height) {
        return create_texture(new_width, new_height, rgba_data.data());

    } else {
        // Reuse existing texture, just upload new data
        return upload_data(rgba_data.data(), width_ * height_ * 4);
    }
}

P10Error ImageTexture::Impl::create_texture(int width, int height, const void* data) {
    // Clear existing texture
    clear();

    width_ = width;
    height_ = height;

    auto result = create_image(width, height);
    if (result.is_error())
        return result;

    result = allocate_memory();
    if (result.is_error())
        return result;

    result = upload_data(data, width * height * 4);
    if (result.is_error())
        return result;

    result = create_image_view();
    if (result.is_error())
        return result;

    result = create_sampler();
    if (result.is_error())
        return result;

    result = create_descriptor_set();
    if (result.is_error())
        return result;

    return P10Error::Ok;
}

P10Error ImageTexture::Impl::create_image(int width, int height) {
    VkImageCreateInfo image_info {};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.extent.width = width;
    image_info.extent.height = height;
    image_info.extent.depth = 1;
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.format = VK_FORMAT_R8G8B8A8_UNORM;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    return wrap_vk_result(vkCreateImage(context_.device, &image_info, nullptr, &image_));
}

P10Error ImageTexture::Impl::allocate_memory() {
    VkMemoryRequirements mem_requirements;
    vkGetImageMemoryRequirements(context_.device, image_, &mem_requirements);

    VkMemoryAllocateInfo alloc_info {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex =
        find_memory_type(mem_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    auto result = wrap_vk_result(vkAllocateMemory(context_.device, &alloc_info, nullptr, &memory_));
    if (result.is_error())
        return result;

    return wrap_vk_result(vkBindImageMemory(context_.device, image_, memory_, 0));
}

P10Error ImageTexture::Impl::upload_data(const void* data, size_t size) {
    // Create staging buffer
    VkBuffer staging_buffer;
    VkDeviceMemory staging_memory;

    VkBufferCreateInfo buffer_info {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    auto result =
        wrap_vk_result(vkCreateBuffer(context_.device, &buffer_info, nullptr, &staging_buffer));
    if (result.is_error())
        return result;

    VkMemoryRequirements mem_requirements;
    vkGetImageMemoryRequirements(context_.device, image_, &mem_requirements);

    VkMemoryAllocateInfo alloc_info {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = size;
    alloc_info.memoryTypeIndex = find_memory_type(
        mem_requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    result =
        wrap_vk_result(vkAllocateMemory(context_.device, &alloc_info, nullptr, &staging_memory));
    if (result.is_error()) {
        vkDestroyBuffer(context_.device, staging_buffer, nullptr);
        return result;
    }

    vkBindBufferMemory(context_.device, staging_buffer, staging_memory, 0);

    // Copy data to staging buffer
    void* mapped_data;
    vkMapMemory(context_.device, staging_memory, 0, size, 0, &mapped_data);
    std::memcpy(mapped_data, data, size);
    vkUnmapMemory(context_.device, staging_memory);

    // Transition image layout and copy data
    VkCommandBuffer cmd_buffer;
    VkCommandBufferAllocateInfo cmd_alloc_info {};
    cmd_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc_info.commandPool = context_.command_pool;
    cmd_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc_info.commandBufferCount = 1;

    vkAllocateCommandBuffers(context_.device, &cmd_alloc_info, &cmd_buffer);

    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(cmd_buffer, &begin_info);

    // Transition to transfer dst
    VkImageMemoryBarrier barrier {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image_;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(
        cmd_buffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &barrier
    );

    // Copy buffer to image
    VkBufferImageCopy region {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {static_cast<uint32_t>(width_), static_cast<uint32_t>(height_), 1};

    vkCmdCopyBufferToImage(
        cmd_buffer,
        staging_buffer,
        image_,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &region
    );

    // Transition to shader read
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(
        cmd_buffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &barrier
    );

    vkEndCommandBuffer(cmd_buffer);

    // Submit and wait
    VkSubmitInfo submit_info {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd_buffer;

    vkQueueSubmit(context_.queue, 1, &submit_info, VK_NULL_HANDLE);
    vkQueueWaitIdle(context_.queue);

    // Cleanup
    vkFreeCommandBuffers(context_.device, context_.command_pool, 1, &cmd_buffer);
    vkDestroyBuffer(context_.device, staging_buffer, nullptr);
    vkFreeMemory(context_.device, staging_memory, nullptr);

    return P10Error::Ok;
}

P10Error ImageTexture::Impl::create_image_view() {
    VkImageViewCreateInfo view_info {};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = image_;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = VK_FORMAT_R8G8B8A8_UNORM;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;

    return wrap_vk_result(vkCreateImageView(context_.device, &view_info, nullptr, &image_view_));
}

P10Error ImageTexture::Impl::create_sampler() {
    VkSamplerCreateInfo sampler_info {};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.anisotropyEnable = VK_FALSE;
    sampler_info.maxAnisotropy = 1.0f;
    sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    sampler_info.unnormalizedCoordinates = VK_FALSE;
    sampler_info.compareEnable = VK_FALSE;
    sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    return wrap_vk_result(vkCreateSampler(context_.device, &sampler_info, nullptr, &sampler_));
}

P10Error ImageTexture::Impl::create_descriptor_set() {
    descriptor_set_ = ImGui_ImplVulkan_AddTexture(
        sampler_,
        image_view_,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    );

    if (descriptor_set_ == VK_NULL_HANDLE) {
        return P10Error::InvalidOperation << "Failed to create ImGui descriptor set";
    }

    return P10Error::Ok;
}

uint32_t
ImageTexture::Impl::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(context_.physical_device, &mem_properties);

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
        if ((type_filter & (1 << i))
            && (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    return 0;
}

void ImageTexture::Impl::clear() {
    if (!is_valid())
        return;

    if (descriptor_set_) {
        ImGui_ImplVulkan_RemoveTexture(descriptor_set_);
        descriptor_set_ = VK_NULL_HANDLE;
    }
    if (sampler_) {
        vkDestroySampler(context_.device, sampler_, nullptr);
        sampler_ = VK_NULL_HANDLE;
    }
    if (image_view_) {
        vkDestroyImageView(context_.device, image_view_, nullptr);
        image_view_ = VK_NULL_HANDLE;
    }
    if (image_) {
        vkDestroyImage(context_.device, image_, nullptr);
        image_ = VK_NULL_HANDLE;
    }
    if (memory_) {
        vkFreeMemory(context_.device, memory_, nullptr);
        memory_ = VK_NULL_HANDLE;
    }

    width_ = 0;
    height_ = 0;
}
}  // namespace p10::guiapp
#pragma once

#include <vector>

#include <ptensor/p10_error.hpp>
#include <vulkan/vulkan.h>

struct SDL_Window;

namespace p10::guiapp {
class GuiAppParameters;

struct VulkanContext {
    std::string vulkan_app_title;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t queue_family = (uint32_t)-1;
    VkQueue queue = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;
    VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;

    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    std::vector<VkImage> swapchain_images;
    std::vector<VkImageView> swapchain_image_views;
    std::vector<VkFramebuffer> framebuffers;
    VkRenderPass render_pass = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> command_buffers;
    VkSemaphore image_available_semaphore = VK_NULL_HANDLE;
    VkSemaphore render_finished_semaphore = VK_NULL_HANDLE;
    VkFence in_flight_fence = VK_NULL_HANDLE;

    int width = 1280;
    int height = 720;
    VkFormat surface_format = VK_FORMAT_B8G8R8A8_UNORM;
};

P10Error initialize_vulkan(VulkanContext& vk, SDL_Window* window, const GuiAppParameters& params);
void cleanup_vulkan(VulkanContext& vk);

}  // namespace p10::guiapp
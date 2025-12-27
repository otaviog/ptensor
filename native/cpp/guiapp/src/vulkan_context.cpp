#include "vulkan_context.hpp"

#include <SDL_vulkan.h>

#include "gui_app_parameters.hpp"
#include "vulkan_error.hpp"

namespace p10::guiapp {

namespace {
    P10Error create_instance(VulkanContext& vk, SDL_Window* window, const GuiAppParameters& params);
    void select_physical_device(VulkanContext& vk);
    P10Error create_device(VulkanContext& vk);
    P10Error create_swapchain(VulkanContext& vk);
    P10Error create_render_pass(VulkanContext& vk);
    P10Error create_framebuffers(VulkanContext& vk);
    P10Error create_command_pool_and_buffers(VulkanContext& vk);
    P10Error create_sync_objects(VulkanContext& vk);
    P10Error create_descriptor_pool(VulkanContext& vk);
}  // namespace

P10Error initialize_vulkan(VulkanContext& vk, SDL_Window* window, const GuiAppParameters& params) {
    auto result = create_instance(vk, window, params);
    if (result.is_error()) {
        goto CLEANUP;
    }

    if (SDL_Vulkan_CreateSurface(window, vk.instance, &vk.surface) != 0) {
        result = P10Error::InvalidOperation << "Failed to create Vulkan surface" << SDL_GetError();
    }
    select_physical_device(vk);
    result = create_device(vk);
    if (result.is_error()) {
        goto CLEANUP;
    }
    result = create_swapchain(vk);
    if (result.is_error()) {
        goto CLEANUP;
    }
    result = create_render_pass(vk);
    if (result.is_error()) {
        goto CLEANUP;
    }
    result = create_framebuffers(vk);
    if (result.is_error()) {
        goto CLEANUP;
    }
    result = create_command_pool_and_buffers(vk);
    if (result.is_error()) {
        goto CLEANUP;
    }
    result = create_sync_objects(vk);
    if (result.is_error()) {
        goto CLEANUP;
    }
    result = create_descriptor_pool(vk);
    if (result.is_error()) {
        goto CLEANUP;
    }

CLEANUP:
    if (result.is_error()) {
        cleanup_vulkan(vk);
    }
    return result;
}

namespace {
    P10Error
    create_instance(VulkanContext& vk, SDL_Window* window, const GuiAppParameters& params) {
        VkApplicationInfo app_info {};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        vk.vulkan_app_title = params.title();
        app_info.pApplicationName = vk.vulkan_app_title.c_str();
        app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.pEngineName = "No Engine";
        app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.apiVersion = VK_API_VERSION_1_0;

        // Get required extensions from SDL
        unsigned int sdl_extension_count;
        SDL_Vulkan_GetInstanceExtensions(window, &sdl_extension_count, nullptr);
        std::vector<const char*> extensions(sdl_extension_count);
        SDL_Vulkan_GetInstanceExtensions(window, &sdl_extension_count, extensions.data());

        VkInstanceCreateInfo create_info {};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.pApplicationInfo = &app_info;
        create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        create_info.ppEnabledExtensionNames = extensions.data();
        create_info.enabledLayerCount = 0;

        return wrap_vk_result(vkCreateInstance(&create_info, nullptr, &vk.instance));
    }

    void select_physical_device(VulkanContext& vk) {
        uint32_t device_count = 0;
        vkEnumeratePhysicalDevices(vk.instance, &device_count, nullptr);
        std::vector<VkPhysicalDevice> devices(device_count);
        vkEnumeratePhysicalDevices(vk.instance, &device_count, devices.data());

        vk.physical_device = devices[0];

        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(vk.physical_device, &queue_family_count, nullptr);
        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(
            vk.physical_device,
            &queue_family_count,
            queue_families.data()
        );

        for (uint32_t i = 0; i < queue_family_count; i++) {
            if (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                vk.queue_family = i;
                break;
            }
        }
    }

    P10Error create_device(VulkanContext& vk) {
        VkDeviceQueueCreateInfo queue_info {};
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = vk.queue_family;
        queue_info.queueCount = 1;
        float queue_priority = 1.0f;
        queue_info.pQueuePriorities = &queue_priority;

        const char* device_extensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

        VkDeviceCreateInfo create_info {};
        create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        create_info.queueCreateInfoCount = 1;
        create_info.pQueueCreateInfos = &queue_info;
        create_info.enabledExtensionCount = 1;
        create_info.ppEnabledExtensionNames = device_extensions;

        P10_RETURN_IF_ERROR(
            wrap_vk_result(vkCreateDevice(vk.physical_device, &create_info, nullptr, &vk.device))
        );

        vkGetDeviceQueue(vk.device, vk.queue_family, 0, &vk.queue);
        return P10Error::Ok;
    }

    P10Error create_swapchain(VulkanContext& vk) {
        VkSurfaceCapabilitiesKHR capabilities;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk.physical_device, vk.surface, &capabilities);

        vk.width = capabilities.currentExtent.width;
        vk.height = capabilities.currentExtent.height;

        VkSwapchainCreateInfoKHR create_info {};
        create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        create_info.surface = vk.surface;
        create_info.minImageCount = capabilities.minImageCount + 1;
        create_info.imageFormat = vk.surface_format;
        create_info.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        create_info.imageExtent = capabilities.currentExtent;
        create_info.imageArrayLayers = 1;
        create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        create_info.preTransform = capabilities.currentTransform;
        create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        create_info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        create_info.clipped = VK_TRUE;

        P10_RETURN_IF_ERROR(
            wrap_vk_result(vkCreateSwapchainKHR(vk.device, &create_info, nullptr, &vk.swapchain))
        );

        uint32_t image_count;
        P10_RETURN_IF_ERROR(
            wrap_vk_result(vkGetSwapchainImagesKHR(vk.device, vk.swapchain, &image_count, nullptr))
        );
        vk.swapchain_images.resize(image_count);
        return wrap_vk_result(vkGetSwapchainImagesKHR(
            vk.device,
            vk.swapchain,
            &image_count,
            vk.swapchain_images.data()
        ));
    }

    P10Error create_render_pass(VulkanContext& vk) {
        VkAttachmentDescription color_attachment {};
        color_attachment.format = vk.surface_format;
        color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
        color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference color_attachment_ref {};
        color_attachment_ref.attachment = 0;
        color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &color_attachment_ref;

        VkSubpassDependency dependency {};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo render_pass_info {};
        render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        render_pass_info.attachmentCount = 1;
        render_pass_info.pAttachments = &color_attachment;
        render_pass_info.subpassCount = 1;
        render_pass_info.pSubpasses = &subpass;
        render_pass_info.dependencyCount = 1;
        render_pass_info.pDependencies = &dependency;
        return wrap_vk_result(
            vkCreateRenderPass(vk.device, &render_pass_info, nullptr, &vk.render_pass)
        );
    }

    P10Error create_framebuffers(VulkanContext& vk) {
        vk.swapchain_image_views.resize(vk.swapchain_images.size());
        vk.framebuffers.resize(vk.swapchain_images.size());

        for (size_t i = 0; i < vk.swapchain_images.size(); i++) {
            VkImageViewCreateInfo view_info {};
            view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            view_info.image = vk.swapchain_images[i];
            view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view_info.format = vk.surface_format;
            view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            view_info.subresourceRange.baseMipLevel = 0;
            view_info.subresourceRange.levelCount = 1;
            view_info.subresourceRange.baseArrayLayer = 0;
            view_info.subresourceRange.layerCount = 1;

            P10_RETURN_IF_ERROR(wrap_vk_result(
                vkCreateImageView(vk.device, &view_info, nullptr, &vk.swapchain_image_views[i])
            ));

            VkFramebufferCreateInfo framebuffer_info {};
            framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebuffer_info.renderPass = vk.render_pass;
            framebuffer_info.attachmentCount = 1;
            framebuffer_info.pAttachments = &vk.swapchain_image_views[i];
            framebuffer_info.width = vk.width;
            framebuffer_info.height = vk.height;
            framebuffer_info.layers = 1;

            P10_RETURN_IF_ERROR(wrap_vk_result(
                vkCreateFramebuffer(vk.device, &framebuffer_info, nullptr, &vk.framebuffers[i])
            ));
        }

        return P10Error::Ok;
    }

    P10Error create_command_pool_and_buffers(VulkanContext& vk) {
        VkCommandPoolCreateInfo pool_info {};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = vk.queue_family;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        P10_RETURN_IF_ERROR(
            wrap_vk_result(vkCreateCommandPool(vk.device, &pool_info, nullptr, &vk.command_pool))
        );

        vk.command_buffers.resize(vk.framebuffers.size());

        VkCommandBufferAllocateInfo alloc_info {};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = vk.command_pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = (uint32_t)vk.command_buffers.size();

        return wrap_vk_result(
            vkAllocateCommandBuffers(vk.device, &alloc_info, vk.command_buffers.data())
        );
    }

    P10Error create_sync_objects(VulkanContext& vk) {
        VkSemaphoreCreateInfo semaphore_info {};
        semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fence_info {};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        P10_RETURN_IF_ERROR(wrap_vk_result(
            vkCreateSemaphore(vk.device, &semaphore_info, nullptr, &vk.image_available_semaphore)
        ));
        P10_RETURN_IF_ERROR(wrap_vk_result(
            vkCreateSemaphore(vk.device, &semaphore_info, nullptr, &vk.render_finished_semaphore)
        ));

        return wrap_vk_result(vkCreateFence(vk.device, &fence_info, nullptr, &vk.in_flight_fence));
    }

    P10Error create_descriptor_pool(VulkanContext& vk) {
        VkDescriptorPoolSize pool_sizes[] = {
            {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
            {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
            {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
            {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}
        };

        VkDescriptorPoolCreateInfo pool_info {};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool_info.maxSets = 1000 * 11;
        pool_info.poolSizeCount = 11;
        pool_info.pPoolSizes = pool_sizes;

        return wrap_vk_result(
            vkCreateDescriptorPool(vk.device, &pool_info, nullptr, &vk.descriptor_pool)
        );
    }
}  // namespace

void cleanup_vulkan(VulkanContext& vk) {
    if (vk.device) {
        vkDeviceWaitIdle(vk.device);

        if (vk.image_available_semaphore) {
            vkDestroySemaphore(vk.device, vk.image_available_semaphore, nullptr);
        }

        if (vk.render_finished_semaphore) {
            vkDestroySemaphore(vk.device, vk.render_finished_semaphore, nullptr);
        }
        if (vk.in_flight_fence) {
            vkDestroyFence(vk.device, vk.in_flight_fence, nullptr);
        }
        if (vk.command_pool) {
            vkDestroyCommandPool(vk.device, vk.command_pool, nullptr);
        }

        for (auto framebuffer : vk.framebuffers) {
            if (framebuffer) {
                vkDestroyFramebuffer(vk.device, framebuffer, nullptr);
            }
        }

        for (auto image_view : vk.swapchain_image_views) {
            if (image_view) {
                vkDestroyImageView(vk.device, image_view, nullptr);
            }
        }

        if (vk.render_pass) {
            vkDestroyRenderPass(vk.device, vk.render_pass, nullptr);
        }

        if (vk.swapchain) {
            vkDestroySwapchainKHR(vk.device, vk.swapchain, nullptr);
        }
        if (vk.descriptor_pool) {
            vkDestroyDescriptorPool(vk.device, vk.descriptor_pool, nullptr);
        }
        if (vk.surface) {
            vkDestroySurfaceKHR(vk.instance, vk.surface, nullptr);
        }
        vkDestroyDevice(vk.device, nullptr);
    }

    if (vk.instance) {
        vkDestroyInstance(vk.instance, nullptr);
    }
}

}  // namespace p10::guiapp
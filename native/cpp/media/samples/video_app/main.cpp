// PTensor Media GUI Sample Application
// Demonstrates video frame display with ImGui + Vulkan + SDL2

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>
#include <SDL.h>
#include <SDL_vulkan.h>
#include <imgui.h>
#include <vulkan/vulkan.h>
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"
#include <ptensor/media/io/media_capture.hpp>

// Vulkan data
struct VulkanContext {
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

static void check_vk_result(VkResult err) {
    if (err != VK_SUCCESS) {
        std::cerr << "Vulkan Error: " << err << std::endl;
        std::abort();
    }
}

void create_instance(VulkanContext& vk, SDL_Window* window) {
    VkApplicationInfo app_info {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "PTensor Video Viewer";
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

    check_vk_result(vkCreateInstance(&create_info, nullptr, &vk.instance));
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

void create_device(VulkanContext& vk) {
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

    check_vk_result(vkCreateDevice(vk.physical_device, &create_info, nullptr, &vk.device));
    vkGetDeviceQueue(vk.device, vk.queue_family, 0, &vk.queue);
}

void create_swapchain(VulkanContext& vk) {
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

    check_vk_result(vkCreateSwapchainKHR(vk.device, &create_info, nullptr, &vk.swapchain));

    uint32_t image_count;
    vkGetSwapchainImagesKHR(vk.device, vk.swapchain, &image_count, nullptr);
    vk.swapchain_images.resize(image_count);
    vkGetSwapchainImagesKHR(vk.device, vk.swapchain, &image_count, vk.swapchain_images.data());
}

void create_render_pass(VulkanContext& vk) {
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

    check_vk_result(vkCreateRenderPass(vk.device, &render_pass_info, nullptr, &vk.render_pass));
}

void create_framebuffers(VulkanContext& vk) {
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

        check_vk_result(
            vkCreateImageView(vk.device, &view_info, nullptr, &vk.swapchain_image_views[i])
        );

        VkFramebufferCreateInfo framebuffer_info {};
        framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebuffer_info.renderPass = vk.render_pass;
        framebuffer_info.attachmentCount = 1;
        framebuffer_info.pAttachments = &vk.swapchain_image_views[i];
        framebuffer_info.width = vk.width;
        framebuffer_info.height = vk.height;
        framebuffer_info.layers = 1;

        check_vk_result(
            vkCreateFramebuffer(vk.device, &framebuffer_info, nullptr, &vk.framebuffers[i])
        );
    }
}

void create_command_pool_and_buffers(VulkanContext& vk) {
    VkCommandPoolCreateInfo pool_info {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = vk.queue_family;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    check_vk_result(vkCreateCommandPool(vk.device, &pool_info, nullptr, &vk.command_pool));

    vk.command_buffers.resize(vk.framebuffers.size());

    VkCommandBufferAllocateInfo alloc_info {};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = vk.command_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = (uint32_t)vk.command_buffers.size();

    check_vk_result(vkAllocateCommandBuffers(vk.device, &alloc_info, vk.command_buffers.data()));
}

void create_sync_objects(VulkanContext& vk) {
    VkSemaphoreCreateInfo semaphore_info {};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fence_info {};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    check_vk_result(
        vkCreateSemaphore(vk.device, &semaphore_info, nullptr, &vk.image_available_semaphore)
    );
    check_vk_result(
        vkCreateSemaphore(vk.device, &semaphore_info, nullptr, &vk.render_finished_semaphore)
    );
    check_vk_result(vkCreateFence(vk.device, &fence_info, nullptr, &vk.in_flight_fence));
}

void create_descriptor_pool(VulkanContext& vk) {
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

    check_vk_result(vkCreateDescriptorPool(vk.device, &pool_info, nullptr, &vk.descriptor_pool));
}

void cleanup_vulkan(VulkanContext& vk) {
    vkDeviceWaitIdle(vk.device);

    vkDestroySemaphore(vk.device, vk.image_available_semaphore, nullptr);
    vkDestroySemaphore(vk.device, vk.render_finished_semaphore, nullptr);
    vkDestroyFence(vk.device, vk.in_flight_fence, nullptr);

    vkDestroyCommandPool(vk.device, vk.command_pool, nullptr);

    for (auto framebuffer : vk.framebuffers) {
        vkDestroyFramebuffer(vk.device, framebuffer, nullptr);
    }

    for (auto image_view : vk.swapchain_image_views) {
        vkDestroyImageView(vk.device, image_view, nullptr);
    }

    vkDestroyRenderPass(vk.device, vk.render_pass, nullptr);
    vkDestroySwapchainKHR(vk.device, vk.swapchain, nullptr);
    vkDestroyDescriptorPool(vk.device, vk.descriptor_pool, nullptr);
    vkDestroySurfaceKHR(vk.instance, vk.surface, nullptr);
    vkDestroyDevice(vk.device, nullptr);
    vkDestroyInstance(vk.instance, nullptr);
}

int main(int argc, char** argv) {
    CLI::App app{"PTensor Video Viewer - ImGui + Vulkan + SDL2"};

    std::string video_path;
    app.add_option("video", video_path, "Path to video file")
        ->required()
        ->check(CLI::ExistingFile);

    int window_width = 1280;
    int window_height = 720;
    app.add_option("-W,--width", window_width, "Window width")
        ->default_val(1280);
    app.add_option("-H,--height", window_height, "Window height")
        ->default_val(720);

    bool fullscreen = false;
    app.add_flag("-f,--fullscreen", fullscreen, "Start in fullscreen mode");

    CLI11_PARSE(app, argc, argv);

    // Check and set SDL video driver
    // const char* sdl_video_driver = std::getenv("SDL_VIDEODRIVER");
    // if (!sdl_video_driver) {
    //     // Try to use x11 or wayland
    //     std::cout << "SDL_VIDEODRIVER not set. Attempting to use x11...\n";
    //     setenv("SDL_VIDEODRIVER", "x11", 0);
    // } else {
    //     std::cout << "Using SDL_VIDEODRIVER: " << sdl_video_driver << "\n";
    // }

    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cerr << "SDL_Init Error: " << SDL_GetError() << std::endl;
        std::cerr << "\nTroubleshooting:\n";
        std::cerr << "1. Make sure you have a display server running (X11 or Wayland)\n";
        std::cerr << "2. Set DISPLAY environment variable (e.g., export DISPLAY=:0)\n";
        std::cerr << "3. For WSL2, install VcXsrv or X410 on Windows\n";
        std::cerr << "4. Set SDL_VIDEODRIVER=x11 or SDL_VIDEODRIVER=wayland\n";
        return 1;
    }

    // Check if Vulkan is available
    if (!SDL_Vulkan_LoadLibrary(nullptr)) {
        std::cerr << "Vulkan support not available: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return 1;
    }

    // Create window
    Uint32 window_flags = SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE;
    if (fullscreen) {
        window_flags |= SDL_WINDOW_FULLSCREEN_DESKTOP;
    }

    SDL_Window* window = SDL_CreateWindow(
        "PTensor Video Viewer",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        window_width,
        window_height,
        window_flags
    );

    if (!window) {
        std::cerr << "SDL_CreateWindow Error: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return 1;
    }

    // Initialize Vulkan
    VulkanContext vk;
    create_instance(vk, window);
    SDL_Vulkan_CreateSurface(window, vk.instance, &vk.surface);
    select_physical_device(vk);
    create_device(vk);
    create_swapchain(vk);
    create_render_pass(vk);
    create_framebuffers(vk);
    create_command_pool_and_buffers(vk);
    create_sync_objects(vk);
    create_descriptor_pool(vk);

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    ImGui_ImplSDL2_InitForVulkan(window);
    ImGui_ImplVulkan_InitInfo init_info {};
    init_info.Instance = vk.instance;
    init_info.PhysicalDevice = vk.physical_device;
    init_info.Device = vk.device;
    init_info.QueueFamily = vk.queue_family;
    init_info.Queue = vk.queue;
    init_info.DescriptorPool = vk.descriptor_pool;
    init_info.MinImageCount = 2;
    init_info.ImageCount = (uint32_t)vk.swapchain_images.size();
    init_info.CheckVkResultFn = check_vk_result;
    init_info.RenderPass = vk.render_pass;

    ImGui_ImplVulkan_Init(&init_info);

    // Upload fonts
    ImGui_ImplVulkan_CreateFontsTexture();
    ImGui_ImplVulkan_DestroyFontsTexture();

    // Open video file
    auto result = p10::media::MediaCapture::open_file(video_path);
    if (!result.is_ok()) {
        std::cerr << "Error opening video: " << result.error() << std::endl;
        return 1;
    }

    auto capture = result.unwrap();
    const auto params = capture.get_parameters();
    const auto video_params = params.video_parameters();

    std::cout << "Loaded video: " << video_params.width() << "x" << video_params.height() << " @ "
              << video_params.frame_rate().to_double() << " fps\n";

    // Main loop
    bool running = true;
    bool playing = false;
    int current_frame = 0;
    p10::media::VideoFrame frame;

    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT) {
                running = false;
            }
        }

        // Start ImGui frame
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        // UI
        ImGui::Begin("Video Controls");
        ImGui::Text("Video: %s", video_path.c_str());
        ImGui::Text("Resolution: %zux%zu", video_params.width(), video_params.height());
        ImGui::Text("Frame Rate: %.2f fps", video_params.frame_rate().to_double());
        ImGui::Separator();

        if (ImGui::Button(playing ? "Pause" : "Play")) {
            playing = !playing;
        }
        ImGui::SameLine();
        if (ImGui::Button("Next Frame")) {
            auto err = capture.next_frame();
            if (err == p10::P10Error::Ok) {
                err = capture.get_video(frame);
                if (err == p10::P10Error::Ok) {
                    current_frame++;
                }
            }
        }

        ImGui::Text("Current Frame: %d", current_frame);
        if (!frame.as_bytes().empty()) {
            ImGui::Text("Frame Time: %.3f s", frame.get_time().to_seconds());
        }

        ImGui::End();

        // Render ImGui
        ImGui::Render();

        vkWaitForFences(vk.device, 1, &vk.in_flight_fence, VK_TRUE, UINT64_MAX);
        vkResetFences(vk.device, 1, &vk.in_flight_fence);

        uint32_t image_index;
        vkAcquireNextImageKHR(
            vk.device,
            vk.swapchain,
            UINT64_MAX,
            vk.image_available_semaphore,
            VK_NULL_HANDLE,
            &image_index
        );

        VkCommandBufferBeginInfo cmd_begin_info {};
        cmd_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cmd_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        VkCommandBuffer command_buffer = vk.command_buffers[image_index];
        vkResetCommandBuffer(command_buffer, 0);
        vkBeginCommandBuffer(command_buffer, &cmd_begin_info);

        VkRenderPassBeginInfo render_pass_begin_info {};
        render_pass_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        render_pass_begin_info.renderPass = vk.render_pass;
        render_pass_begin_info.framebuffer = vk.framebuffers[image_index];
        render_pass_begin_info.renderArea.extent.width = vk.width;
        render_pass_begin_info.renderArea.extent.height = vk.height;
        VkClearValue clear_color = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
        render_pass_begin_info.clearValueCount = 1;
        render_pass_begin_info.pClearValues = &clear_color;

        vkCmdBeginRenderPass(command_buffer, &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), command_buffer);
        vkCmdEndRenderPass(command_buffer);
        vkEndCommandBuffer(command_buffer);

        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo render_submit_info {};
        render_submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        render_submit_info.waitSemaphoreCount = 1;
        render_submit_info.pWaitSemaphores = &vk.image_available_semaphore;
        render_submit_info.pWaitDstStageMask = &wait_stage;
        render_submit_info.commandBufferCount = 1;
        render_submit_info.pCommandBuffers = &command_buffer;
        render_submit_info.signalSemaphoreCount = 1;
        render_submit_info.pSignalSemaphores = &vk.render_finished_semaphore;

        vkQueueSubmit(vk.queue, 1, &render_submit_info, vk.in_flight_fence);

        VkPresentInfoKHR present_info {};
        present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present_info.waitSemaphoreCount = 1;
        present_info.pWaitSemaphores = &vk.render_finished_semaphore;
        present_info.swapchainCount = 1;
        present_info.pSwapchains = &vk.swapchain;
        present_info.pImageIndices = &image_index;

        vkQueuePresentKHR(vk.queue, &present_info);
    }

    // Cleanup
    capture.close();
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    cleanup_vulkan(vk);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}

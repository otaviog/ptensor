#include "gui_app.impl.hpp"

#include <SDL.h>
#include <SDL_vulkan.h>

#include "image_texture.impl.hpp"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"
#include "vulkan_context.hpp"

namespace p10::guiapp {
GuiApp::Impl::Impl(GuiApp& parent) : parent_(parent) {}

GuiApp::Impl::~Impl() {
    if (running_) {
        quit();
    }
}

P10Error GuiApp::Impl::start(const GuiAppParameters& params) {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        return P10Error::InvalidOperation << SDL_GetError();
    }

    if (SDL_Vulkan_LoadLibrary(nullptr) != 0) {
        const std::string error_msg =
            "Vulkan support not available: " + std::string(SDL_GetError());
        SDL_Quit();
        return P10Error::InvalidOperation << error_msg;
    }
    Uint32 window_flags = SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE;

    params_ = params;
    window_ = SDL_CreateWindow(
        params.title().c_str(),
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        params.width(),
        params.height(),
        window_flags
    );
    if (!window_) {
        const std::string error_msg = "SDL_CreateWindow Error: " + std::string(SDL_GetError());
        SDL_Quit();
        return P10Error::InvalidOperation << error_msg;
    }

    vk_.reset(new VulkanContext);
    if (initialize_vulkan(*vk_, window_, params).is_error()) {
        const std::string error_msg = "Failed to initialize Vulkan: " + std::string(SDL_GetError());
        SDL_DestroyWindow(window_);
        SDL_Quit();
        return P10Error::InvalidOperation << error_msg;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    ImGui_ImplSDL2_InitForVulkan(window_);
    ImGui_ImplVulkan_InitInfo init_info {};
    init_info.Instance = vk_->instance;
    init_info.PhysicalDevice = vk_->physical_device;
    init_info.Device = vk_->device;
    init_info.QueueFamily = vk_->queue_family;
    init_info.Queue = vk_->queue;
    init_info.DescriptorPool = vk_->descriptor_pool;
    init_info.MinImageCount = 2;
    init_info.ImageCount = (uint32_t)vk_->swapchain_images.size();
    // init_info.CheckVkResultFn = check_vk_result;
    init_info.RenderPass = vk_->render_pass;

    ImGui_ImplVulkan_Init(&init_info);

    // Upload fonts
    ImGui_ImplVulkan_CreateFontsTexture();
    ImGui_ImplVulkan_DestroyFontsTexture();

    parent_.on_initialize();
    main_loop();
    return P10Error::Ok;
}

void GuiApp::Impl::main_loop() {
    running_ = true;
    while (running_) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT) {
                running_ = false;
            }
        }

        // Start ImGui frame
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();
        parent_.on_render();
        ImGui::Render();

        vkWaitForFences(vk_->device, 1, &vk_->in_flight_fence, VK_TRUE, UINT64_MAX);
        vkResetFences(vk_->device, 1, &vk_->in_flight_fence);

        uint32_t image_index;
        vkAcquireNextImageKHR(
            vk_->device,
            vk_->swapchain,
            UINT64_MAX,
            vk_->image_available_semaphore,
            VK_NULL_HANDLE,
            &image_index
        );

        VkCommandBufferBeginInfo cmd_begin_info {};
        cmd_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cmd_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        VkCommandBuffer command_buffer = vk_->command_buffers[image_index];
        vkResetCommandBuffer(command_buffer, 0);
        vkBeginCommandBuffer(command_buffer, &cmd_begin_info);

        VkRenderPassBeginInfo render_pass_begin_info {};
        render_pass_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        render_pass_begin_info.renderPass = vk_->render_pass;
        render_pass_begin_info.framebuffer = vk_->framebuffers[image_index];
        render_pass_begin_info.renderArea.extent.width = vk_->width;
        render_pass_begin_info.renderArea.extent.height = vk_->height;
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
        render_submit_info.pWaitSemaphores = &vk_->image_available_semaphore;
        render_submit_info.pWaitDstStageMask = &wait_stage;
        render_submit_info.commandBufferCount = 1;
        render_submit_info.pCommandBuffers = &command_buffer;
        render_submit_info.signalSemaphoreCount = 1;
        render_submit_info.pSignalSemaphores = &vk_->render_finished_semaphore;

        vkQueueSubmit(vk_->queue, 1, &render_submit_info, vk_->in_flight_fence);

        VkPresentInfoKHR present_info {};
        present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present_info.waitSemaphoreCount = 1;
        present_info.pWaitSemaphores = &vk_->render_finished_semaphore;
        present_info.swapchainCount = 1;
        present_info.pSwapchains = &vk_->swapchain;
        present_info.pImageIndices = &image_index;

        vkQueuePresentKHR(vk_->queue, &present_info);
    }
}

ImageTexture GuiApp::Impl::create_texture() {
    ImageTextureVkContext context;
    context.command_pool = vk_->command_pool;
    context.device = vk_->device;
    context.physical_device = vk_->physical_device;
    context.queue = vk_->queue;
    return ImageTexture(new ImageTexture::Impl(context));
}

void GuiApp::Impl::quit() {
    running_ = false;
    parent_.on_cleanup();
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    cleanup_vulkan(*vk_);
    SDL_DestroyWindow(window_);
    SDL_Quit();
}
}  // namespace p10::guiapp
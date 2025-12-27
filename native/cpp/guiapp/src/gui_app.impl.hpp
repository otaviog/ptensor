#pragma once

#include "gui_app.hpp"
#include "gui_app_parameters.hpp"

struct SDL_Window;

namespace p10::guiapp {
struct VulkanContext;

class GuiApp::Impl {
  public:
    Impl(GuiApp& parent);

    ~Impl();

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    Impl(const Impl&&) = delete;
    Impl& operator=(const Impl&&) = delete;

    P10Error start(const GuiAppParameters& params);

    void quit();

  private:
    void main_loop();

    GuiApp& parent_;
    GuiAppParameters params_;
    std::unique_ptr<VulkanContext> vk_;
    SDL_Window* window_ = nullptr;
    std::atomic<bool> running_ = std::atomic<bool>(false);
};
}  // namespace p10::guiapp
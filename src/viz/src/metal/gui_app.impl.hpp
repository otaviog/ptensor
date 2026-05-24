#pragma once

#include <atomic>
#include <memory>

#include "gui_app.hpp"
#include "gui_app_parameters.hpp"
#include "image_texture.hpp"

struct SDL_Window;

namespace p10::viz {

// Defined in gui_app.impl.mm to keep ObjC types out of this header
struct MetalContext;

class GuiApp::Impl {
  public:
    Impl(GuiApp& parent);

    ~Impl();

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    Impl(const Impl&&) = delete;
    Impl& operator=(const Impl&&) = delete;

    P10Error start(const GuiAppParameters& params);

    ImageTexture create_texture();

    void quit();

    void main_loop();

  private:
    GuiApp& parent_;
    GuiAppParameters params_;
    std::unique_ptr<MetalContext> metal_;
    SDL_Window* window_ = nullptr;
    std::atomic<bool> running_{false};
};

}  // namespace p10::viz

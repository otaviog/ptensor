#pragma once

#include <memory>

#include <imgui.h>
#include <ptensor/p10_error.hpp>

#include "image_texture.hpp"

namespace p10::viz {

class GuiAppParameters;
struct VulkanContext;

class GuiApp;

P10Error run_app(GuiApp& app, const GuiAppParameters& params);

/// Base class for ImGui-based applications.
class GuiApp {
  private:
    class Impl;

  public:
    /// Construct an uninitialized GuiApp.
    GuiApp();

    virtual ~GuiApp();

    /// Create a GPU texture for rendering.
    ImageTexture create_texture();

    /// Signal the application to quit.
    void quit();

  protected:
    /// Override to perform initialization when the app starts.
    virtual void on_initialize() {}

    /// Override to render each frame.
    virtual void on_render();

    /// Override to perform cleanup when the app exits.
    virtual void on_cleanup() {}

  private:
    friend P10Error run_app(GuiApp& app, const GuiAppParameters& params);

    P10Error start(const GuiAppParameters& params);

    static ImGuiContext* get_current_context();

    void run();

    std::shared_ptr<Impl> impl_;
};

}  // namespace p10::viz
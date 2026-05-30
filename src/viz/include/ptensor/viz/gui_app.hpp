#pragma once

#include <memory>

#include <imgui.h>
#include <ptensor/p10_error.hpp>

#include "image_texture.hpp"

namespace p10::viz {

class GuiAppParameters;
struct VulkanContext;

class GuiApp;

/// Start a GUI application with the given parameters.
inline P10Error run_app(GuiApp& app, const GuiAppParameters& params);

/// Base class for ImGui-based applications.
class GuiApp {
  private:
    class Impl;

  public:
    /// Construct an uninitialized GuiApp.
    GuiApp();

    /// Create a GPU texture for rendering.
    ImageTexture create_texture();

    /// Signal the application to quit.
    void quit();

    virtual ~GuiApp();

  protected:
    /// Override to perform initialization when the app starts.
    virtual void on_initialize() {}

    /// Override to render each frame.
    virtual void on_render();

    /// Override to perform cleanup when the app exits.
    virtual void on_cleanup() {}

  private:
    friend inline P10Error run_app(GuiApp& app, const GuiAppParameters& params);

    P10Error start(const GuiAppParameters& params);

    static ImGuiContext* get_current_context();

    void run();

    std::shared_ptr<Impl> impl_;
};

/// Start a GUI application with the given parameters.
inline P10Error run_app(GuiApp& app, const GuiAppParameters& params) {
    auto error = app.start(params);
    if (error.is_error()) {
        return error;
    }

    ImGui::SetCurrentContext(GuiApp::get_current_context());
    app.run();
    return P10Error::Ok;
}

}  // namespace p10::viz
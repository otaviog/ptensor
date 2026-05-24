#pragma once

#include <memory>

#include <imgui.h>
#include <ptensor/p10_error.hpp>

#include "image_texture.hpp"

namespace p10::viz {
class GuiAppParameters;
struct VulkanContext;

class GuiApp;
inline P10Error run_app(GuiApp& app, const GuiAppParameters& params);

class GuiApp {
  private:
    class Impl;

  public:
    GuiApp();

    ImageTexture create_texture();

    void quit();

    virtual ~GuiApp();

  protected:
    virtual void on_initialize() {}

    virtual void on_render();

    virtual void on_cleanup() {}

  private:
    friend inline P10Error run_app(GuiApp& app, const GuiAppParameters& params);

    P10Error start(const GuiAppParameters& params);

    static ImGuiContext* get_current_context();

    void run();

    std::shared_ptr<Impl> impl_;
};

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
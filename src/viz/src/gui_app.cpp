#include "gui_app.hpp"

#include <imgui.h>

#include "gui_app.impl.hpp"

namespace p10::viz {

GuiApp::GuiApp() : impl_(new GuiApp::Impl(*this)) {}

GuiApp::~GuiApp() = default;

ImageTexture GuiApp::create_texture() {
    return impl_->create_texture();
}

void GuiApp::quit() {
    impl_->quit();
}

void GuiApp::on_render() {
    static bool my_tool_active = true;
    static float my_color[4] = {0.0f, 0.5f, 0.5f, 1.0f};

    ImGui::Begin("My First Tool", &my_tool_active, ImGuiWindowFlags_MenuBar);
    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Open..", "Ctrl+O")) { /* Do stuff */
            }
            if (ImGui::MenuItem("Save", "Ctrl+S")) { /* Do stuff */
            }
            if (ImGui::MenuItem("Close", "Ctrl+W")) {
                my_tool_active = false;
            }
            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }

    // Edit a color stored as 4 floats
    ImGui::ColorEdit4("Color", my_color);

    // Generate samples and plot them
    float samples[100];
    for (int n = 0; n < 100; n++) {
        samples[n] = sinf(n * 0.2f + float(ImGui::GetTime()) * 1.5f);
    }
    ImGui::PlotLines("Samples", samples, 100);

    // Display contents in a scrolling region
    ImGui::TextColored(ImVec4(1, 1, 0, 1), "Important Stuff");
    ImGui::BeginChild("Scrolling");
    for (int n = 0; n < 50; n++) {
        ImGui::Text("%04d: Some text", n);
    }
    ImGui::EndChild();
    ImGui::End();
}

P10Error GuiApp::start(const GuiAppParameters& params) {
    return impl_->start(params);
}

ImGuiContext* GuiApp::get_current_context() {
    return ImGui::GetCurrentContext();
}

void GuiApp::run() {
    impl_->main_loop();
}

P10Error run_app(GuiApp& app, const GuiAppParameters& params) {
    auto error = app.start(params);
    if (error.is_error()) {
        return error;
    }

    ImGui::SetCurrentContext(GuiApp::get_current_context());
    app.run();
    return P10Error::Ok;
}

}  // namespace p10::viz

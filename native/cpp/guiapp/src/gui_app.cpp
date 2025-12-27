#include "gui_app.hpp"

#include <imgui.h>
#include <vulkan/vulkan.h>

#include "gui_app.impl.hpp"

namespace p10::guiapp {
GuiApp::GuiApp() : impl_(new GuiApp::Impl(*this)) {}

GuiApp::~GuiApp() = default;

P10Error GuiApp::start(const GuiAppParameters& params) {
    return impl_->start(params);
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
    for (int n = 0; n < 100; n++)
        samples[n] = sinf(n * 0.2f + float(ImGui::GetTime()) * 1.5f);
    ImGui::PlotLines("Samples", samples, 100);

    // Display contents in a scrolling region
    ImGui::TextColored(ImVec4(1, 1, 0, 1), "Important Stuff");
    ImGui::BeginChild("Scrolling");
    for (int n = 0; n < 50; n++)
        ImGui::Text("%04d: Some text", n);
    ImGui::EndChild();
    ImGui::End();
}

void GuiApp::quit() {
    on_cleanup();
    impl_->quit();
}
}  // namespace p10::guiapp
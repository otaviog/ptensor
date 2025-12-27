#include <iostream>

#include <ptensor/guiapp/gui_app.hpp>
#include <ptensor/guiapp/gui_app_parameters.hpp>
#include <ptensor/guiapp/image_texture.hpp>
#include <ptensor/io/image.hpp>

class WindowApp: public p10::guiapp::GuiApp {
  protected:
    void on_initialize() override {
        texture_ = create_texture();
        auto image =
            p10::io::load_image("tests/data/image/image01.png").expect("Unable to load test image");
        texture_.upload(image);
    }

    void on_render() override {
        ImGui::Begin("Image Viewer");
        if (texture_.is_valid()) {
            ImGui::Image(
                texture_.texture_id(),
                ImVec2(static_cast<float>(texture_.width()), static_cast<float>(texture_.height()))
            );
        }
        ImGui::End();
    }

    void on_cleanup() override {
        // Cleanup code here
    }

  private:
    p10::guiapp::ImageTexture texture_;
};

int main(int /*argc*/, char** /*argv*/) {
    WindowApp app;

    if (auto status = app.start(p10::guiapp::GuiAppParameters()); status.is_error()) {
        std::cout << status.to_string() << std::endl;
    }

    return 0;
}
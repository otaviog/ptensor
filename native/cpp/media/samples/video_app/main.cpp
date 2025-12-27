// PTensor Media GUI Sample Application
// Demonstrates video frame display with ImGui + Vulkan + SDL2

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>
#include <ptensor/guiapp/gui_app_parameters.hpp>
#include <ptensor/media/io/media_capture.hpp>

#include "video_player_app.hpp"

int main(int argc, char** argv) {
    CLI::App app("PTensor Video Viewer - ImGui + Vulkan + SDL2");

    std::string video_path;
    app.add_option("video", video_path, "Path to video file")->required()->check(CLI::ExistingFile);

    int window_width = 1280;
    int window_height = 720;
    app.add_option("-W,--width", window_width, "Window width")->default_val(1280);
    app.add_option("-H,--height", window_height, "Window height")->default_val(720);

    bool fullscreen = false;
    app.add_flag("-f,--fullscreen", fullscreen, "Start in fullscreen mode");

    CLI11_PARSE(app, argc, argv);

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

    VideoPlayerApp video_app(capture);
    video_app.start(p10::guiapp::GuiAppParameters().title("Simple video player"));

    return 0;
}

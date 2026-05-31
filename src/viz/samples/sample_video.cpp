#include <iostream>

#include <ptensor/io/image.hpp>
#include <ptensor/media/io/media_capture.hpp>
#include <ptensor/viz/gui_app.hpp>
#include <ptensor/viz/gui_app_parameters.hpp>
#include <ptensor/viz/image_texture.hpp>
#include <ptensor/viz/video_player_component.hpp>

class VideoApp: public p10::viz::GuiApp {
  public:
    VideoApp(p10::media::MediaCapture capture) : video_player_ {capture, *this} {}

  protected:
    void on_initialize() override {
        video_player_.on_initialize();
    }

    void on_render() override {
        video_player_.on_render();
    }

    void on_cleanup() override {}

  private:
    p10::viz::VideoPlayerComponent video_player_;
};

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "tests/data/video/test_video.mp4";
    p10::media::MediaCapture capture =
        p10::media::MediaCapture::open_file(path).expect("Unable to open test video");
    VideoApp app {capture};
    if (auto status = run_app(app, p10::viz::GuiAppParameters()); status.is_error()) {
        std::cout << status.to_string() << std::endl;
    }

    return 0;
}
#include "video_player_app.hpp"

#include <iostream>

VideoPlayerApp::VideoPlayerApp(p10::media::MediaCapture capture) : capture_(std::move(capture)) {}

void VideoPlayerApp::on_initialize() {
    video_texture_ = create_texture();
}

void VideoPlayerApp::on_render() {
    p10::media::VideoFrame frame;
    if (auto status = capture_.next_frame(); status.is_error()) {
        std::cerr << status.to_string() << std::endl;
        return;
    }

    auto status = capture_.get_video(frame);
    if (status.is_ok()) {
        if (frame.width() > 0)
        video_texture_.upload(frame.image()).expect("Unable to upload image");
    } else {
        std::cerr << status.to_string() << std::endl;
    }

    ImGui::Begin("Video Player");
    ImGui::Text("Video Player");
    ImGui::Separator();

    // Display video frame
    ImGui::Text("Video Frame:");
    ImGui::Image(
        video_texture_.texture_id(),
        ImVec2(
            static_cast<float>(video_texture_.width()),
            static_cast<float>(video_texture_.height())
        )
    );

    ImGui::End();
}

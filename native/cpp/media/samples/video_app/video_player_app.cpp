#include "video_player_app.hpp"

#include <iostream>

VideoPlayerApp::VideoPlayerApp(p10::media::MediaCapture capture) :
    capture_(std::move(capture)),
    video_frame_rate_(capture_.get_parameters().video_parameters().frame_rate().inverse()) {}

void VideoPlayerApp::on_initialize() {
    video_texture_ = create_texture();
    timer_.start();
}

void VideoPlayerApp::on_render() {
    bool should_grab_next = false;

    if (play_state_ == PlayState::Step) {
        should_grab_next = true;
    } else if (play_state_ == PlayState::Playing) {
        const auto elapsed_time = timer_.elapsed(video_frame_rate_);
        should_grab_next = elapsed_time.stamp() > current_frame_ts_.stamp();
    }

    if (should_grab_next) {
        auto next_frame_res = capture_.next_frame();
        if (next_frame_res.is_ok()) {
            const bool has_next_frame = next_frame_res.unwrap();
            if (has_next_frame) {
                auto status = capture_.get_video(current_frame_);
                if (status.is_ok()) {
                    video_texture_.upload(current_frame_.image()).expect("Unable to upload image");
                    current_frame_ts_ = current_frame_.time();
                } else {
                    std::cerr << status.to_string() << std::endl;
                }
                if (play_state_ == PlayState::Step) {
                    play_state_ = PlayState::Paused;
                }
            } else {
                play_state_ = PlayState::Stopped;
            }
        } else if (next_frame_res.is_error()) {
            play_state_ = PlayState::Stopped;
            return;
        }
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

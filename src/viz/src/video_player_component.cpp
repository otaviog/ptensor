#include "video_player_component.hpp"

#include <iostream>

namespace p10::viz {
VideoPlayerComponent::VideoPlayerComponent(p10::media::MediaCapture capture, GuiApp& app) :
    app_(app),
    capture_(std::move(capture)),
    video_frame_rate_(capture_.get_parameters().video_parameters().frame_rate().inverse()) {}

void VideoPlayerComponent::on_initialize() {
    video_texture_ = app_.create_texture();
}

void VideoPlayerComponent::on_render() {
    update_next_frame();
    ImGui::Begin("Video Player");
    ImGui::Text("Video Player");
    ImGui::Separator();

    // --- Playback Controls ---
    ImGui::BeginDisabled(play_state_ == PlayState::Playing);
    if (ImGui::Button("Play")) {
        play_state_ = PlayState::Playing;
        timer_.start();
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::BeginDisabled(play_state_ != PlayState::Playing);
    if (ImGui::Button("Pause")) {
        play_state_ = PlayState::Paused;
        timer_.pause();
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Step")) {
        play_state_ = PlayState::Step;
        timer_.pause();
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(true);
    if (ImGui::Button("Stop")) {
        play_state_ = PlayState::Stopped;
        timer_.reset();
        // Optionally reset frame/time here if desired
    }
    ImGui::EndDisabled();
    ImGui::Separator();

    // Add your custom drawing commands

    ImGui::Image(
        video_texture_.texture_id(),
        ImVec2(
            static_cast<float>(video_texture_.width()),
            static_cast<float>(video_texture_.height())
        )
    );

    // Draw on top of the image
    on_render_hook(ImGui::GetWindowDrawList(), ImGui::GetItemRectMin(), ImGui::GetItemRectSize());
    ImGui::End();
}

void VideoPlayerComponent::update_next_frame() {
    bool should_grab_next = false;

    if (play_state_ == PlayState::Step) {
        should_grab_next = true;
    } else if (play_state_ == PlayState::Playing) {
        const auto elapsed_time = timer_.elapsed(video_frame_rate_);
        should_grab_next = elapsed_time > current_frame_ts_;
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
                    on_new_frame(current_frame_);
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
}

void VideoPlayerComponent::on_new_frame(p10::media::VideoFrame& frame) {
    for (const auto& callback : new_frame_callback_) {
        callback(frame);
    }
}

void VideoPlayerComponent::on_render_hook(
    ImDrawList* draw_list,
    ImVec2 image_pos,
    ImVec2 image_size
) {
    for (const auto& hook : render_hook_callback_) {
        hook(draw_list, image_pos, image_size);
    }
}
}  // namespace p10::viz

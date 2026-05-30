#include "video_player_component.hpp"
#include <iostream>

#include "logging.hpp"
#include "ptensor/media/io/media_capture.hpp"

namespace p10::viz {

VideoPlayerComponent::VideoPlayerComponent(p10::media::MediaCapture capture, GuiApp& app) :
    app_(app),

    capture_(std::move(capture)),
    video_frame_rate_(capture_.get_parameters().video_parameters().frame_rate().inverse()) {
    if (capture_.is_stream()) {
        play_state_ = PlayState::Streaming;
    }
}

void VideoPlayerComponent::on_initialize() {
    video_texture_ = app_.create_texture();
}

void VideoPlayerComponent::on_render() {
    update_next_frame();
    ImGui::Begin("Video Player");
    ImGui::Text("Video Player");
    ImGui::Separator();

    // --- Playback Controls ---
    button_play();
    ImGui::SameLine();
    button_pause();
    ImGui::SameLine();
    button_step();
    ImGui::SameLine();
    button_stop();
    ImGui::Separator();

    image_camera();
    ImGui::End();
}

void VideoPlayerComponent::button_play() {
    ImGui::BeginDisabled(play_state_ == PlayState::Playing);
    if (ImGui::Button("Play")) {
        play_state_ = PlayState::Playing;
        timer_.start();
    }
    ImGui::EndDisabled();
}

void VideoPlayerComponent::button_pause() {
    ImGui::BeginDisabled(play_state_ != PlayState::Playing);
    if (ImGui::Button("Pause")) {
        play_state_ = PlayState::Paused;
        timer_.pause();
    }
    ImGui::EndDisabled();
}

void VideoPlayerComponent::button_step() {
    if (ImGui::Button("Step")) {
        play_state_ = PlayState::Step;
        timer_.pause();
    }
}

void VideoPlayerComponent::button_stop() {
    ImGui::BeginDisabled(true);
    if (ImGui::Button("Stop")) {
        play_state_ = PlayState::Stopped;
        timer_.reset();
        capture_.seek(0.0);
    }
    ImGui::EndDisabled();
}

void VideoPlayerComponent::image_camera() const {
    ImGui::Image(
        video_texture_.texture_id(),
        ImVec2(
            static_cast<float>(video_texture_.width()),
            static_cast<float>(video_texture_.height())
        )
    );
    on_render_hook(ImGui::GetWindowDrawList(), ImGui::GetItemRectMin(), ImGui::GetItemRectSize());
}

void VideoPlayerComponent::update_next_frame() {
    if (capture_.is_stream()) {
        capture_next_frame();
    } else {
        sync_next_frame();
    }
}

void VideoPlayerComponent::capture_next_frame() {
    using enum PlayState;
    auto next_frame_res = capture_.next_frame(media::MediaCapture::Poll);
    if (next_frame_res.is_error()) {
        LOGGER.error(next_frame_res);
        play_state_ = Stopped;
        return;
    }
    if (next_frame_res.unwrap() == p10::media::MediaCapture::NotReady) {
        return;  // no frame queued yet; keep displaying the last one
    }
    if (next_frame_res.unwrap() == p10::media::MediaCapture::Done) {
        play_state_ = Stopped;
        return;
    }

    if (auto status = capture_.get_video(current_frame_); status.is_ok()) {
        video_texture_.upload(current_frame_.image()).expect("Unable to upload image");
        current_frame_ts_ = current_frame_.time();
        on_new_frame(current_frame_);
        play_state_ = Streaming;
    } else {
        std::cerr << status.to_string() << "\n";
    }
}

void VideoPlayerComponent::sync_next_frame() {
    using enum PlayState;
    bool should_grab_next = false;

    if (play_state_ == Step) {
        should_grab_next = true;
    } else if (play_state_ == Playing) {
        const auto elapsed_time = timer_.elapsed(video_frame_rate_);
        should_grab_next = elapsed_time > current_frame_ts_;
    }

    if (!should_grab_next) {
        return;
    }

    auto next_frame_res = capture_.next_frame();
    if (next_frame_res.is_error()) {
        play_state_ = Stopped;
        return;
    }
    switch (next_frame_res.unwrap()) {
        case p10::media::MediaCapture::NotReady:
            return;
        case p10::media::MediaCapture::Done:
            play_state_ = Stopped;
            return;
        case p10::media::MediaCapture::Available:
            break;
    }

    if (auto status = capture_.get_video(current_frame_); status.is_ok()) {
        video_texture_.upload(current_frame_.image()).expect("Unable to upload image");
        current_frame_ts_ = current_frame_.time();
        on_new_frame(current_frame_);
        if (play_state_ == Step) {
            play_state_ = Paused;
        }
    } else {
        std::cerr << status.to_string() << "\n";
    }
}

void VideoPlayerComponent::on_new_frame(p10::media::VideoFrame& frame) const {
    for (const auto& callback : new_frame_callback_) {
        callback(frame);
    }
}

void VideoPlayerComponent::on_render_hook(
    ImDrawList* draw_list,
    ImVec2 image_pos,
    ImVec2 image_size
) const {
    for (const auto& hook : render_hook_callback_) {
        hook(draw_list, image_pos, image_size);
    }
}
}  // namespace p10::viz

#pragma once

#include <vector>

#include <ptensor/media/io/media_capture.hpp>
#include <ptensor/media/video_frame.hpp>

#include "gui_app.hpp"
#include "image_texture.hpp"
#include "timer.hpp"

namespace p10::viz {
class VideoPlayerComponent {
  public:
    using NewFrameCallback = std::function<void(p10::media::VideoFrame& frame)>;

    using RenderHookCallback = std::function<void(ImDrawList*, ImVec2, ImVec2)>;

    enum class PlayState { Playing, Paused, Stopped, Step };

    VideoPlayerComponent(media::MediaCapture capture, GuiApp& app);

    void on_initialize();

    void on_render();

    void add_new_frame_callback(NewFrameCallback callback) {
        new_frame_callback_.push_back(callback);
    }

    void add_render_hook_callback(RenderHookCallback callback) {
        render_hook_callback_.push_back(callback);
    }

  protected:
    void on_new_frame(p10::media::VideoFrame& frame);

    void on_render_hook(ImDrawList* draw_list, ImVec2 image_pos, ImVec2 image_size);

  private:
    void update_next_frame();

    GuiApp& app_;

    media::MediaCapture capture_;
    media::Time current_frame_ts_;
    media::Rational video_frame_rate_;
    ImageTexture video_texture_;
    SystemTimer timer_;

    PlayState play_state_ {PlayState::Stopped};
    media::VideoFrame current_frame_;

    std::vector<NewFrameCallback> new_frame_callback_;
    std::vector<RenderHookCallback> render_hook_callback_;
};
}  // namespace p10::viz

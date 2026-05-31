#pragma once

#include <vector>

#include <imgui.h>
#include <ptensor/media/io/media_capture.hpp>
#include <ptensor/media/video_frame.hpp>

#include "gui_app.hpp"
#include "image_texture.hpp"
#include "timer.hpp"

namespace p10::viz {

struct ToComponentSpace {
    ImVec2 pos;
    ImVec2 component_size;
    ImVec2 frame_size;

    ImVec2 operator()(int x, int y) const {
        return {
            pos.x + (x / frame_size.x) * component_size.x,
                pos.y + (y / frame_size.y) * component_size.y,
        };
    }
};

/// Playback state for video player.
enum class PlayState { Playing, Paused, Stopped, Step, Streaming };

/// Video player UI component with playback controls and frame callbacks.
class VideoPlayerComponent {
  public:
    /// Callback invoked when a new frame is available.
    using NewFrameCallback = std::function<void(p10::media::VideoFrame& frame)>;

    /// Callback for custom rendering over the video image.
    using RenderHookCallback = std::function<void(ImDrawList*, ToComponentSpace)>;

    /// Construct a video player component.
    VideoPlayerComponent(media::MediaCapture capture, GuiApp& app);

    /// Initialize video playback resources.
    void on_initialize();

    /// Render the video player UI and update playback.
    void on_render();

    /// Register a callback to be invoked when a new frame arrives.
    void add_new_frame_callback(const NewFrameCallback& callback) {
        new_frame_callback_.push_back(callback);
    }

    /// Register a callback for custom rendering over the video.
    void add_render_hook_callback(const RenderHookCallback& callback) {
        render_hook_callback_.push_back(callback);
    }

  protected:
    /// Invoke all registered new-frame callbacks.
    void on_new_frame(p10::media::VideoFrame& frame) const;

    /// Invoke all registered render-hook callbacks.
    void on_render_hook(ImDrawList* draw_list, ImVec2 image_pos, ImVec2 image_size) const;

  private:
    void button_play();

    void button_pause();

    void button_step();

    void button_stop();

    void image_camera() const;

    void update_next_frame();

    void capture_next_frame();

    void sync_next_frame();

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

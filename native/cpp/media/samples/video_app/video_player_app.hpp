#pragma once

#include <ptensor/guiapp/gui_app.hpp>
#include <ptensor/guiapp/image_texture.hpp>
#include <ptensor/media/io/media_capture.hpp>

#include "timer.hpp"

class VideoPlayerApp: public p10::guiapp::GuiApp {
  public:
    enum class PlayState { Playing, Paused, Stopped, Step };

    VideoPlayerApp(p10::media::MediaCapture capture);

  protected:
    void on_initialize() override;
    void on_render() override;

  private:
    p10::media::MediaCapture capture_;
    p10::media::Time current_frame_ts_;
    p10::media::Rational video_frame_rate_;

    p10::guiapp::ImageTexture video_texture_;
    Timer<> timer_;

    PlayState play_state_ {PlayState::Playing};
    
                
    p10::media::VideoFrame current_frame_;
};

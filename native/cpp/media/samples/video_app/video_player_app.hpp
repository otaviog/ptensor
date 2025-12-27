#pragma once

#include <ptensor/guiapp/gui_app.hpp>
#include <ptensor/guiapp/image_texture.hpp>
#include <ptensor/media/io/media_capture.hpp>

class VideoPlayerApp: public p10::guiapp::GuiApp {
  public:
    VideoPlayerApp(p10::media::MediaCapture capture);

  protected:
    void on_initialize() override;
    void on_render() override;

  private:
    p10::media::MediaCapture capture_;
    p10::guiapp::ImageTexture video_texture_;
};
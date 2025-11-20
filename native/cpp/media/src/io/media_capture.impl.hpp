#pragma once
#include <ptensor/p10_result.hpp>

#include "io/media_capture.hpp"
#include "video_frame.hpp"

namespace p10::media {
class MediaCapture::Impl {
  public:
    virtual MediaParameters get_parameters() const = 0;

    virtual P10Error next_frame() = 0;

    virtual P10Result<VideoFrame> get_video() = 0;

    virtual P10Result<AudioFrame> get_audio() = 0;
};
}  // namespace p10::media
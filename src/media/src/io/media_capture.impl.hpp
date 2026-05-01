#pragma once
#include <optional>

#include <ptensor/p10_result.hpp>

#include "io/media_capture.hpp"

namespace p10::media {
class VideoFrame;
class AudioFrame;

class MediaCapture::Impl {
  public:
    virtual ~Impl() = default;

    virtual void close() = 0;

    virtual MediaParameters get_parameters() const = 0;

    virtual P10Result<bool> next_frame() = 0;

    virtual std::optional<int64_t> video_frame_count() const = 0;

    virtual P10Error get_video(VideoFrame& frame) = 0;

    virtual P10Error get_audio(AudioFrame& frame) = 0;

    virtual std::optional<double> duration() const = 0;
};
}  // namespace p10::media
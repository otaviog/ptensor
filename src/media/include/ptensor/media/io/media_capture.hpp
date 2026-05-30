#pragma once
#include <memory>
#include <optional>
#include <string>

#include <ptensor/p10_result.hpp>

#include "../media_parameters.hpp"

namespace p10::media {
class VideoFrame;
class AudioFrame;

constexpr int NO_DEVICE_SELECTED = -1;
    
class MediaCapture {
  public:
    
    class Impl;

    MediaCapture() = default;

    static P10Result<MediaCapture> open_file(const std::string& path);

    static P10Result<MediaCapture> open_stream(int audio_device_index = NO_DEVICE_SELECTED, int video_device_index = NO_DEVICE_SELECTED);

    void close();

    MediaParameters get_parameters() const;

    P10Result<bool> next_frame();

    P10Error get_video(VideoFrame& frame);

    P10Error get_audio(AudioFrame& frame);

    std::optional<int64_t> video_frame_count() const;

    std::optional<double> duration() const;

  private:
    explicit MediaCapture(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}

    std::shared_ptr<Impl> impl_;
};
}  // namespace p10::media

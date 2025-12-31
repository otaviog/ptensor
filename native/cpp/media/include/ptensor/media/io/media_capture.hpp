#pragma once
#include <memory>
#include <string>

#include <ptensor/p10_result.hpp>

#include "../audio_frame.hpp"
#include "../media_parameters.hpp"
#include "../video_frame.hpp"

namespace p10::media {
class MediaCapture {
  public:
    class Impl;

    static P10Result<MediaCapture> open_file(const std::string& path);

    void close();

    MediaParameters get_parameters() const;

    P10Result<bool> next_frame();

    P10Error get_video(VideoFrame& frame);

    P10Error get_audio(AudioFrame& frame);

  private:
    explicit MediaCapture(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}

    std::shared_ptr<Impl> impl_;
};
}  // namespace p10::media
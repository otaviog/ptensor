#pragma once

#include <memory>
#include <string>

#include <ptensor/media/media_parameters.hpp>
#include <ptensor/p10_error.hpp>
#include <ptensor/p10_result.hpp>

namespace p10::media {
class VideoFrame;
class AudioFrame;

class MediaWriter {
  public:
    class Impl;
    static P10Result<MediaWriter> open_file(const std::string& path, const MediaParameters& params);

    void close();

    MediaParameters get_parameters() const;

    P10Error write_video(const VideoFrame& frame);

    P10Error write_audio(const AudioFrame& frame);

  private:
    explicit MediaWriter(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}

    std::shared_ptr<Impl> impl_;
};

}  // namespace p10::media

#pragma once

#include <memory>
#include <string>

namespace p10::media {
class MediaParameters;
class VideoFrame;
class AudioFrame;

class MediaWriter {
  public:
    class Impl;
    static P10Result<MediaWriter>
    create_file(const std::string& path, const MediaParameters& params);

    void close();

    MediaParameters get_parameters() const;

    P10Error write_video(const VideoFrame& frame);

    P10Error write_audio(const AudioFrame& frame);

  private:
    explicit MediaWriter(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}

    std::shared_ptr<Impl> impl_;
};

}  // namespace p10::media
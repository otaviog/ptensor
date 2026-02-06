#pragma once

#include <memory>

#include <libavformat/avformat.h>

#include "../media_writer.impl.hpp"

struct AVFormatContext;

namespace p10::media {

class FfmpegMediaWriter: public MediaWriter::Impl {
  public:
    static P10Result<std::shared_ptr<FfmpegMediaWriter>>
    create(const std::string& path, const MediaParameters& params);

    ~FfmpegMediaWriter() override;

    void close() override;

    MediaParameters get_parameters() const override;

    P10Error write_video(const VideoFrame& frame) override;

    P10Error write_audio(const AudioFrame& frame) override;

  private:
    AVFormatContext* format_context_ = nullptr;
};
}  // namespace p10::media

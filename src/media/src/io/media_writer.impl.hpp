#pragma once

#include "io/media_writer.hpp"

namespace p10::media {
class MediaWriter::Impl {
  public:
    Impl() = default;
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&) = delete;
    Impl& operator=(Impl&&) = delete;
    virtual ~Impl() = default;

    virtual void close() = 0;

    virtual MediaParameters get_parameters() const = 0;

    virtual P10Error write_video(const VideoFrame& frame) = 0;

    virtual P10Error write_audio(const AudioFrame& frame) = 0;
};
}  // namespace p10::media

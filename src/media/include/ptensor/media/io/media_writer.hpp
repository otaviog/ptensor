#pragma once

#include <memory>
#include <string>

#include <ptensor/media/media_parameters.hpp>
#include <ptensor/p10_error.hpp>
#include <ptensor/p10_result.hpp>

namespace p10::media {
class VideoFrame;
class AudioFrame;
class Text;

/// Media file writer for video and audio.
class MediaWriter {
  public:
    class Impl;
    /// Open a media file for writing.
    static P10Result<MediaWriter> open_file(const std::string& path, const MediaParameters& params);

    /// Close the media writer.
    void close();

    /// Get current media parameters.
    MediaParameters get_parameters() const;

    /// Write a video frame.
    P10Error write_video(const VideoFrame& frame);

    /// Write an audio frame.
    P10Error write_audio(const AudioFrame& frame);

    /// Write a text entry to the text stream at `stream_index`.
    ///
    /// The text streams must have been declared via MediaParameters before
    /// open_file(); `stream_index` selects among them in declaration order.
    P10Error write_text(size_t stream_index, const Text& text);

  private:
    explicit MediaWriter(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}

    std::shared_ptr<Impl> impl_;
};

}  // namespace p10::media

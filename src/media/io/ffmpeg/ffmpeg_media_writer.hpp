#pragma once

#include <memory>
#include <vector>

#include "../media_writer.impl.hpp"
#include "ffmpeg_audio_encoder.hpp"
#include "ffmpeg_text_encoder.hpp"
#include "ffmpeg_video_encoder.hpp"

struct AVFormatContext;

namespace p10::media {

class FfmpegMediaWriter: public MediaWriter::Impl {
  public:
    static P10Result<std::shared_ptr<FfmpegMediaWriter>>
    create(const std::string& path, const MediaParameters& params);

    FfmpegMediaWriter(const FfmpegMediaWriter&) = delete;
    FfmpegMediaWriter& operator=(const FfmpegMediaWriter&) = delete;
    FfmpegMediaWriter(FfmpegMediaWriter&&) = delete;
    FfmpegMediaWriter& operator=(FfmpegMediaWriter&&) = delete;

    ~FfmpegMediaWriter() override;

    void close() override;

    MediaParameters get_parameters() const override;

    P10Error write_video(const VideoFrame& frame) override;

    P10Error write_audio(const AudioFrame& frame) override;

    P10Error write_text(size_t stream_index, const Text& text) override;

  private:
    FfmpegMediaWriter(
        AVFormatContext* format_context,
        MediaParameters params,
        std::unique_ptr<FfmpegVideoEncoder> video_encoder,
        std::unique_ptr<FfmpegAudioEncoder> audio_encoder,
        std::vector<std::unique_ptr<FfmpegTextEncoder>> text_encoders
    );

    P10Error flush_video_encoder();
    P10Error flush_audio_encoder();
    P10Error write_video_packet(AVPacket* packet);
    P10Error pop_video_packets();
    P10Error write_audio_packet(AVPacket* packet);

    AVFormatContext* format_context_ = nullptr;
    MediaParameters params_;
    bool header_written_ = false;

    std::unique_ptr<FfmpegVideoEncoder> video_encoder_;
    std::unique_ptr<FfmpegAudioEncoder> audio_encoder_;
    std::vector<std::unique_ptr<FfmpegTextEncoder>> text_encoders_;
};
}  // namespace p10::media

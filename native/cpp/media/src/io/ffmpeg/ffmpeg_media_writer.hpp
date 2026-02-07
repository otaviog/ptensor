#pragma once

#include <memory>

#include "../media_writer.impl.hpp"
#include "ffmpeg_audio_encoder.hpp"
#include "ffmpeg_video_encoder.hpp"

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
    FfmpegMediaWriter(
        AVFormatContext* format_context,
        const MediaParameters& params,
        std::unique_ptr<FfmpegVideoEncoder> video_encoder,
        std::unique_ptr<FfmpegAudioEncoder> audio_encoder
    );

    P10Error flush_video_encoder();
    P10Error flush_audio_encoder();
    P10Error write_video_packet(AVPacket* packet);
    P10Error write_audio_packet(AVPacket* packet);

    AVFormatContext* format_context_ = nullptr;
    MediaParameters params_;
    bool header_written_ = false;
    bool closed_ = false;

    std::unique_ptr<FfmpegVideoEncoder> video_encoder_;
    std::unique_ptr<FfmpegAudioEncoder> audio_encoder_;

    int64_t video_pts_ = 0;
};
}  // namespace p10::media

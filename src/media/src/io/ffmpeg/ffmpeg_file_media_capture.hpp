#pragma once

#include <cstdint>
#include <memory>

#include "ffmpeg_media_capture_engine.hpp"

namespace p10::media {

/// File-based capture. Reads from a path FFmpeg auto-detects, and reports the
/// bounded lifecycle (duration and total frame count) that files have but live
/// devices do not.
class FfmpegFileMediaCapture: public FfmpegMediaCaptureEngine {
  public:
    static P10Result<std::shared_ptr<FfmpegFileMediaCapture>> open(const std::string& path);

    std::optional<int64_t> video_frame_count() const override;

    std::optional<double> duration() const override;

    P10Error seek(double seconds) override;

  private:
    FfmpegFileMediaCapture(
        AVFormatContext* format_ctx,
        std::shared_ptr<FfmpegAudioDecoder> audio_decoder,
        std::shared_ptr<FfmpegVideoDecoder> video_decoder
    ) :
        FfmpegMediaCaptureEngine(format_ctx, std::move(audio_decoder), std::move(video_decoder)) {}
};

}  // namespace p10::media

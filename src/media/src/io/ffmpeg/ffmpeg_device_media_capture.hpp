#pragma once

#include <memory>
#include <optional>
#include <utility>

#include <ptensor/media/audio_parameters.hpp>
#include <ptensor/media/video_parameters.hpp>

#include "ffmpeg_media_capture_engine.hpp"

namespace p10::media {

/// Live device capture (camera / microphone).
///
/// Differs from file capture in three ways that justify a separate type:
///   * Open path: it selects a platform input format (avfoundation / v4l2 /
///     dshow), builds the device URL from indices and translates VideoParameters
///     and AudioParameters into demuxer options (framerate, video_size, etc.).
///   * Lifecycle: a live stream is unbounded, so duration() and
///     video_frame_count() always return nullopt.
///   * Error modes: opening surfaces device-not-found / busy / unsupported
///     capability as IoError rather than file IoError.
class FfmpegDeviceMediaCapture: public FfmpegMediaCaptureEngine {
  public:
    /// Open the selected devices with optional capability configuration.
    /// Pass std::nullopt for index -1 (no device of that kind). VideoParameters
    /// and AudioParameters fields are translated into FFmpeg demuxer options;
    /// zero/default fields are skipped.
    static P10Result<std::shared_ptr<FfmpegDeviceMediaCapture>> open(
        std::optional<std::pair<int, VideoParameters>> video,
        std::optional<std::pair<int, AudioParameters>> audio
    );

    std::optional<int64_t> video_frame_count() const override {
        return std::nullopt;
    }

    std::optional<double> duration() const override {
        return std::nullopt;
    }

  private:
    FfmpegDeviceMediaCapture(
        AVFormatContext* format_ctx,
        std::shared_ptr<FfmpegAudioDecoder> audio_decoder,
        std::shared_ptr<FfmpegVideoDecoder> video_decoder
    ) :
        FfmpegMediaCaptureEngine(format_ctx, std::move(audio_decoder), std::move(video_decoder)) {}
};

}  // namespace p10::media

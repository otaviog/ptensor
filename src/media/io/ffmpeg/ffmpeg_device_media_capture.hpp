#pragma once

#include <memory>
#include <optional>
#include <utility>

#include <ptensor/media/audio_parameters.hpp>
#include <ptensor/media/video_parameters.hpp>

#include "../camera_controls.hpp"
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

    P10Result<int> get_camera_control(CameraControlId id) const override;
    P10Error set_camera_control(CameraControlId id, int value) override;
    P10Result<CameraControlRange> get_camera_control_range(CameraControlId id) const override;
    P10Result<bool> get_camera_auto_control(CameraAutoControlId id) const override;
    P10Error set_camera_auto_control(CameraAutoControlId id, bool enabled) override;

  private:
    FfmpegDeviceMediaCapture(
        AVFormatContext* format_ctx,
        std::shared_ptr<FfmpegAudioDecoder> audio_decoder,
        std::shared_ptr<FfmpegVideoDecoder> video_decoder,
        std::unique_ptr<CameraControlBackend> camera_controls
    ) :
        FfmpegMediaCaptureEngine(format_ctx, std::move(audio_decoder), std::move(video_decoder)),
        camera_controls_(std::move(camera_controls)) {}

    std::unique_ptr<CameraControlBackend> camera_controls_;
};

}  // namespace p10::media

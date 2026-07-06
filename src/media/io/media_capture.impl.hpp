#pragma once
#include <optional>

#include <ptensor/p10_result.hpp>

#include "camera_controls.hpp"
#include "io/media_capture.hpp"
#include "text_streams.hpp"

namespace p10::media {
class VideoFrame;
class AudioFrame;

class MediaCapture::Impl {
  public:
    Impl() = default;
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&) = delete;
    Impl& operator=(Impl&&) = delete;
    virtual ~Impl() = default;

    virtual void close() = 0;

    virtual MediaParameters get_parameters() const = 0;

    virtual P10Result<MediaCapture::NextFrameResult> next_frame(MediaCapture::WaitMode wait) = 0;

    virtual std::optional<int64_t> video_frame_count() const = 0;

    virtual P10Error get_video(VideoFrame& frame) = 0;

    virtual P10Error get_audio(AudioFrame& frame) = 0;

    virtual P10Result<TextStreams> get_text_streams() const {
        return Ok(TextStreams());
    }

    virtual std::optional<double> duration() const = 0;

    virtual P10Error seek(double /*seconds*/) {
        return P10Error::NotImplemented << "Seek not supported for this capture source";
    }

    /// Camera property controls (focus, exposure, brightness, ...). Defaults
    /// to NotImplemented; only live device captures override these.
    virtual P10Result<int> get_camera_control(CameraControlId /*id*/) const {
        return Err(
            P10Error::NotImplemented << "Camera controls not supported for this capture source"
        );
    }

    virtual P10Error set_camera_control(CameraControlId /*id*/, int /*value*/) {
        return P10Error::NotImplemented << "Camera controls not supported for this capture source";
    }

    virtual P10Result<CameraControlRange> get_camera_control_range(CameraControlId /*id*/) const {
        return Err(
            P10Error::NotImplemented << "Camera controls not supported for this capture source"
        );
    }

    virtual P10Result<bool> get_camera_auto_control(CameraAutoControlId /*id*/) const {
        return Err(
            P10Error::NotImplemented << "Camera controls not supported for this capture source"
        );
    }

    virtual P10Error set_camera_auto_control(CameraAutoControlId /*id*/, bool /*enabled*/) {
        return P10Error::NotImplemented << "Camera controls not supported for this capture source";
    }
};
}  // namespace p10::media